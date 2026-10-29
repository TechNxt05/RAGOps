import dataclasses
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List, Optional
from pydantic import BaseModel
from app.db import get_session
from app.models.rag import Document, Chunk
from app.models.user import User, UserRole
from app.auth.access import require_project
from app.auth.deps import get_current_user
from app.rag.engine import RAGEngine

router = APIRouter(prefix="/inspector", tags=["inspector"])

class ChunkDTO(BaseModel):
    id: int
    content: str
    token_count: int = 0  # Placeholder if we don't store it explicitly yet

class DebugSearchResult(BaseModel):
    chunk_id: int
    content: str
    score: float
    document_name: str

class DebugSearchResponse(BaseModel):
    results: List[DebugSearchResult]
    query_analysis: Optional[dict] = None
    pipeline_trace: Optional[dict] = None

class DebugSearchRequest(BaseModel):
    project_id: int
    query: str
    top_k: int = 4
    similarity_threshold: float = 0.0

@router.get("/documents/{document_id}/chunks", response_model=List[ChunkDTO])
def get_document_chunks(
    document_id: int, 
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    require_project(session, current_user, doc.project_id)

    chunks = session.exec(select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index, Chunk.id)).all()
    # Simple estimation for tokens if not stored: chars / 4
    return [ChunkDTO(id=c.id, content=c.content, token_count=len(c.content)//4) for c in chunks]

@router.post("/search", response_model=DebugSearchResponse)
def debug_search(
    request: DebugSearchRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    require_project(session, current_user, request.project_id)

    engine = RAGEngine(session)
    results = engine.search(
        query=request.query,
        project_id=request.project_id,
        k=request.top_k,
        score_threshold=request.similarity_threshold
    )
    output = [
        DebugSearchResult(
            chunk_id=int(doc.metadata.get("chunk_id") or 0),
            content=doc.page_content,
            score=round(float(score), 4),  # calibrated relevance in [0, 1]
            document_name=str(doc.metadata.get("source", "Unknown")).replace("\\", "/").split("/")[-1],
        )
        for doc, score in results
    ]
    analysis = getattr(results, "query_analysis", None)
    if analysis is not None and dataclasses.is_dataclass(analysis):
        analysis = json.loads(json.dumps(dataclasses.asdict(analysis), default=lambda o: getattr(o, "value", str(o))))
    return DebugSearchResponse(
        results=output,
        query_analysis=analysis,
        pipeline_trace=json.loads(json.dumps(getattr(results, "pipeline_trace", None), default=str)),
    )
