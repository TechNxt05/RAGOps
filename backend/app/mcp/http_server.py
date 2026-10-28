"""
Hosted MCP server (Streamable HTTP) at /mcp.

Any MCP client (Claude Desktop, Claude Code, Cursor, custom agents) can connect
with a project API key:

    URL:     https://<your-backend>/mcp/
    Header:  Authorization: Bearer rgo_...

The key pins the session to one project; every tool operates on that project.
"""

import contextvars
import json
import logging
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from sqlmodel import Session, select
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.db import engine
from app.models.rag import Document, ExternalDatabase, Project

logger = logging.getLogger(__name__)

_project_id: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("mcp_project_id", default=None)

mcp = FastMCP(
    name="RAGOps",
    instructions=(
        "Tools for one RAGOps project knowledge base. Use `ask` for grounded answers with "
        "citations, `search_documents` for raw passages, and `query_database` for live data "
        "from the project's connected SQL databases."
    ),
    streamable_http_path="/",
    stateless_http=True,
    json_response=True,
    # Requests are authenticated by API key below, and the public hostname is not
    # known in advance (Cloudflare Tunnel / custom domain), so host pinning is off.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


def _pid() -> int:
    pid = _project_id.get()
    if pid is None:
        raise PermissionError("Missing project API key")
    return pid


@mcp.tool()
async def ask(question: str) -> Dict[str, Any]:
    """Answer a question from the project's documents using the full RAG pipeline
    (hybrid retrieval, reranking, confidence gate). Returns the answer with cited sources."""
    from app.rag.headless import answer_question
    with Session(engine) as db:
        return await answer_question(db, _pid(), question)


@mcp.tool()
async def search_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Retrieve the most relevant passages for a query (no generation)."""
    from app.rag.engine import RAGEngine

    def _run():
        with Session(engine) as db:
            eng = RAGEngine(db)
            docs = eng._single_hybrid_search(query, _pid_value, max(1, min(top_k, 20)))
            from app.services.reranker_service import reranker_service
            ranked, scored = reranker_service.rerank_scored(query, docs, max(1, min(top_k, 20)))
            return [
                {"source": d.metadata.get("source"), "doc_id": d.metadata.get("doc_id"),
                 "chunk_id": d.metadata.get("chunk_id"), "score": round(float(s), 4), "content": d.page_content}
                for d, s in ranked
            ]

    _pid_value = _pid()
    return {"results": await run_in_threadpool(_run)}


@mcp.tool()
async def list_documents(limit: int = 100) -> Dict[str, Any]:
    """List the active documents in the project."""
    pid = _pid()
    with Session(engine) as db:
        docs = db.exec(
            select(Document).where(Document.project_id == pid).where(Document.is_active == True)  # noqa: E712
            .order_by(Document.uploaded_at.desc()).limit(max(1, min(limit, 500)))
        ).all()
        return {"documents": [
            {"id": d.id, "filename": d.filename, "status": d.processing_status, "chunks": d.chunk_count,
             "uploaded_at": d.uploaded_at.isoformat()} for d in docs
        ]}


@mcp.tool()
async def list_databases() -> Dict[str, Any]:
    """List the SQL databases connected to this project and the tables you may query."""
    pid = _pid()
    with Session(engine) as db:
        rows = db.exec(select(ExternalDatabase).where(ExternalDatabase.project_id == pid).where(ExternalDatabase.is_active == True)).all()  # noqa: E712
        return {"databases": [
            {"id": r.id, "name": r.name or r.db_name, "type": r.db_type, "description": r.description,
             "tables": r.allowed_tables or []} for r in rows
        ]}


@mcp.tool()
async def query_database(question: str, database_id: Optional[int] = None) -> Dict[str, Any]:
    """Answer a question from live data in a connected SQL database (read-only).
    Returns the answer, the SQL that was run, and the result rows."""
    from app.rag.database_routes import ask_database
    from app.services.db_connector.sql_agent import DatabaseQueryError
    pid = _pid()
    with Session(engine) as db:
        q = select(ExternalDatabase).where(ExternalDatabase.project_id == pid).where(ExternalDatabase.is_active == True)  # noqa: E712
        if database_id is not None:
            q = q.where(ExternalDatabase.id == database_id)
        ext = db.exec(q).first()
        if ext is None:
            return {"error": "No connected database found for this project"}
        try:
            result = await ask_database(db, ext, question)
        except DatabaseQueryError as exc:
            return {"error": str(exc)}
        result["rows"] = result["rows"][:100]
        return result


class APIKeyAuth:
    """ASGI middleware: resolve the project API key and pin the request to it."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        auth = headers.get("authorization", "")
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else headers.get("x-api-key", "").strip()

        from app.services.api_keys import resolve_key

        def _resolve():
            with Session(engine) as db:
                record = resolve_key(db, key)
                if record and db.get(Project, record.project_id):
                    return record.project_id
                return None

        project_id = await run_in_threadpool(_resolve) if key else None
        if project_id is None:
            response = JSONResponse({"error": "Invalid or missing API key"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
            return await response(scope, receive, send)
        token = _project_id.set(project_id)
        try:
            await self.app(scope, receive, send)
        finally:
            _project_id.reset(token)


def build_app() -> ASGIApp:
    return APIKeyAuth(mcp.streamable_http_app())
