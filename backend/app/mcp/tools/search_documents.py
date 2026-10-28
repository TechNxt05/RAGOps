from pydantic import BaseModel, Field
from sqlmodel import Session
from starlette.concurrency import run_in_threadpool

from app.db import engine
from app.mcp.schemas import MCPContext
from app.rag.engine import RAGEngine


class SearchDocumentsInput(BaseModel):
    query: str = Field(..., description="The query string to search for")
    top_k: int = Field(default=4, description="Number of results to return")


def _search(args: SearchDocumentsInput, project_id: int) -> dict:
    with Session(engine) as session:
        rag_engine = RAGEngine(session)
        config = rag_engine._config_or_default(project_id)
        results = rag_engine.search(
            query=args.query, project_id=project_id, k=args.top_k, score_threshold=config.similarity_threshold,
        )
        return {
            "chunks": [
                {"content": doc.page_content, "score": score, "source": doc.metadata.get("source"),
                 "doc_id": doc.metadata.get("doc_id")}
                for doc, score in results
            ],
            "chunks_before_pruning": getattr(results, "chunks_before_pruning", 0),
            "chunks_after_pruning": getattr(results, "chunks_after_pruning", 0),
            "pruning_reduction_pct": getattr(results, "pruning_reduction_pct", 0.0),
            "used_hybrid_search": getattr(results, "used_hybrid_search", False),
        }


async def search_documents_tool(args: SearchDocumentsInput, context: MCPContext):
    if not context.project_id:
        return {"error": "Project ID required for search"}
    return await run_in_threadpool(_search, args, context.project_id)
