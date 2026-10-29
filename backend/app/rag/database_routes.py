"""
External database connections: a manager connects a company database to a
project and chooses which tables to expose; project members then ask questions
in plain language and get answers computed from live data.
"""

import json
import logging
import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from starlette.concurrency import run_in_threadpool

from app.auth.access import can_write_project, require_not_demo_project, require_project
from app.auth.deps import get_current_user
from app.core.config import settings
from app.core.rate_limit import rate_limiter
from app.db import get_session
from app.models.query_log import QueryLog
from app.models.rag import ExternalDatabase, RAGConfig
from app.models.user import User
from app.services import llm_factory
from app.services.db_connector import sql_agent
from app.services.db_connector.connection_parser import ConnectionParser
from app.services.db_connector.credential_vault import credential_vault

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag/projects/{project_id}/databases", tags=["databases"])


class DatabaseCreate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=128)
    description: Optional[str] = Field(default=None, max_length=2000)
    connection_string: Optional[str] = None
    db_type: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    db_name: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_mode: Optional[str] = "prefer"
    allowed_tables: List[str] = Field(default_factory=list)


class DatabaseUpdate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=128)
    description: Optional[str] = Field(default=None, max_length=2000)
    allowed_tables: Optional[List[str]] = None
    is_active: Optional[bool] = None


class AskBody(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    model_provider: Optional[str] = None
    model_name: Optional[str] = None


def _view(ext: ExternalDatabase, manager: bool) -> dict:
    data = {
        "id": ext.id, "name": ext.name or ext.db_name, "description": ext.description,
        "db_type": ext.db_type, "is_active": ext.is_active, "allowed_tables": ext.allowed_tables or [],
        "updated_at": ext.updated_at,
    }
    if manager:
        data.update({
            "host": ext.host, "port": ext.port, "db_name": ext.db_name, "username": ext.username,
            "ssl_mode": ext.ssl_mode,
            "tables": {t: cols for t, cols in ((ext.schema_cache or {}).get("tables") or {}).items()},
        })
    return data


def _get(session: Session, project_id: int, db_id: int) -> ExternalDatabase:
    ext = session.get(ExternalDatabase, db_id)
    if not ext or ext.project_id != project_id:
        raise HTTPException(status_code=404, detail="Database connection not found")
    return ext


def _connect(ext: ExternalDatabase, password: str):
    return sql_agent.connect(ext.db_type, ext.host, ext.port, ext.db_name, ext.username, password, ext.ssl_mode)


@router.get("")
def list_databases(project_id: int, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    project = require_project(session, user, project_id)
    manager = can_write_project(session, user, project)
    rows = session.exec(select(ExternalDatabase).where(ExternalDatabase.project_id == project_id)).all()
    return [_view(r, manager) for r in rows if manager or r.is_active]


@router.post("")
async def add_database(project_id: int, body: DatabaseCreate, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    project = require_project(session, user, project_id, write=True)
    require_not_demo_project(session, project, "Connecting databases")
    if not settings.secret_key_configured:
        raise HTTPException(status_code=400, detail="Set SECRET_KEY on the server before connecting databases.")

    fields = body.model_dump()
    if body.connection_string:
        parsed = ConnectionParser().parse_connection_string(body.connection_string)
        if not parsed.get("success"):
            raise HTTPException(status_code=400, detail="Could not parse the connection string")
        for k, v in parsed["parsed"].items():
            if k != "raw_uri" and not fields.get(k):
                fields[k] = v
    missing = [k for k in ("db_type", "host", "db_name", "username") if not fields.get(k)]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing: {', '.join(missing)}")
    db_type = fields["db_type"].lower()
    db_type = "postgresql" if db_type in ("postgres", "postgresql") else db_type
    if db_type not in sql_agent.SUPPORTED_DB_TYPES:
        raise HTTPException(status_code=400, detail=f"Supported databases: {', '.join(sql_agent.SUPPORTED_DB_TYPES)}")
    port = int(fields.get("port") or (5432 if db_type == "postgresql" else 3306))

    def _test():
        conn = sql_agent.connect(db_type, fields["host"], port, fields["db_name"], fields["username"],
                                 fields.get("password") or "", fields.get("ssl_mode"))
        try:
            return sql_agent.introspect(conn, db_type)
        finally:
            conn.close()

    try:
        tables = await run_in_threadpool(_test)
    except sql_agent.DatabaseQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    unknown = [t for t in body.allowed_tables if t not in tables]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown tables: {', '.join(unknown)}")
    ext = ExternalDatabase(
        project_id=project_id, name=body.name or fields["db_name"], description=body.description,
        db_type=db_type, host=fields["host"], port=port, db_name=fields["db_name"], username=fields["username"],
        encrypted_credentials=credential_vault.encrypt(fields.get("password") or ""),
        ssl_mode=fields.get("ssl_mode") or "prefer",
        schema_cache={"tables": tables, "refreshed_at": datetime.utcnow().isoformat()},
        allowed_tables=body.allowed_tables,
    )
    session.add(ext)
    session.commit()
    session.refresh(ext)
    return _view(ext, True)


@router.patch("/{db_id}")
def update_database(project_id: int, db_id: int, body: DatabaseUpdate, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    require_project(session, user, project_id, write=True)
    ext = _get(session, project_id, db_id)
    data = body.model_dump(exclude_unset=True)
    if "allowed_tables" in data:
        known = set(((ext.schema_cache or {}).get("tables") or {}).keys())
        unknown = [t for t in data["allowed_tables"] if t not in known]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown tables: {', '.join(unknown)}")
    for k, v in data.items():
        setattr(ext, k, v)
    ext.updated_at = datetime.utcnow()
    session.add(ext)
    session.commit()
    session.refresh(ext)
    return _view(ext, True)


@router.post("/{db_id}/refresh-schema")
async def refresh_schema(project_id: int, db_id: int, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    require_project(session, user, project_id, write=True)
    ext = _get(session, project_id, db_id)
    password = credential_vault.decrypt(ext.encrypted_credentials)

    def _run():
        conn = _connect(ext, password)
        try:
            return sql_agent.introspect(conn, ext.db_type)
        finally:
            conn.close()

    try:
        tables = await run_in_threadpool(_run)
    except sql_agent.DatabaseQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    ext.schema_cache = {"tables": tables, "refreshed_at": datetime.utcnow().isoformat()}
    ext.allowed_tables = [t for t in (ext.allowed_tables or []) if t in tables]
    ext.updated_at = datetime.utcnow()
    session.add(ext)
    session.commit()
    return _view(ext, True)


@router.delete("/{db_id}")
def delete_database(project_id: int, db_id: int, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    require_project(session, user, project_id, write=True)
    session.delete(_get(session, project_id, db_id))
    session.commit()
    return {"ok": True}


async def ask_database(session: Session, ext: ExternalDatabase, question: str, preferred: Optional[dict] = None) -> dict:
    config = session.exec(
        select(RAGConfig).where(RAGConfig.project_id == ext.project_id).where(RAGConfig.is_active == True)  # noqa: E712
    ).first()
    llm, _ = llm_factory.build_with_fallbacks(llm_factory.model_chain(config, preferred), temperature=0.0, max_tokens=1024)
    password = credential_vault.decrypt(ext.encrypted_credentials)
    conn = await run_in_threadpool(_connect, ext, password)
    try:
        return await sql_agent.ask(llm, ext.db_type, conn, (ext.schema_cache or {}).get("tables") or {},
                                   ext.allowed_tables or [], question, ext.description)
    finally:
        conn.close()


@router.post("/{db_id}/ask")
async def ask(project_id: int, db_id: int, body: AskBody, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    require_project(session, user, project_id)
    rate_limiter.check(f"chat:{user.id}", settings.rate_limit_chat_per_min)
    ext = _get(session, project_id, db_id)
    if not ext.is_active:
        raise HTTPException(status_code=400, detail="This database connection is disabled")
    preferred = {"provider": body.model_provider, "name": body.model_name} if body.model_name else None
    t0 = time.time()
    try:
        result = await ask_database(session, ext, body.question, preferred)
    except sql_agent.DatabaseQueryError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    latency_ms = (time.time() - t0) * 1000.0
    session.add(QueryLog(
        project_id=project_id, user_id=user.id, query_text=body.question, response_text=result["answer"],
        model_used=f"database/{ext.db_type}", latency_ms=latency_ms, chunks_retrieved=result["row_count"],
        adaptive_backend_used="live_database",
        pipeline_trace={"database_id": ext.id, "sql": result["sql"], "row_count": result["row_count"],
                        "tables_considered": result["tables_considered"], "attempts": result["attempts"]},
    ))
    session.commit()
    return {**result, "latency_ms": round(latency_ms, 1), "database": ext.name or ext.db_name}
