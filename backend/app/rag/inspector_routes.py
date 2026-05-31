from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List, Optional
from pydantic import BaseModel
from app.db import get_session
from app.models.rag import Document, Chunk
from app.models.user import User
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
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    chunks = session.exec(select(Chunk).where(Chunk.document_id == document_id)).all()
    # Simple estimation for tokens if not stored: chars / 4
    return [ChunkDTO(id=c.id, content=c.content, token_count=len(c.content)//4) for c in chunks]

@router.post("/search", response_model=DebugSearchResponse)
def debug_search(
    request: DebugSearchRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        engine = RAGEngine(session)
        # We use the internal search method directly
        # search returns list of (Document, score)
        results = engine.search(
            query=request.query,
            project_id=request.project_id,
            k=request.top_k,
            score_threshold=request.similarity_threshold
        )
        
        output = []
        for doc, score in results:
            filename = doc.metadata.get("source", "Unknown")
            if "/" in filename or "\\" in filename:
                filename = filename.split("/")[-1].split("\\")[-1]

            # Normalize Score for UI (FAISS L2 Distance -> Similarity)
            try:
                normalized_score = 1 / (1 + score)
            except:
                normalized_score = 0.0

            output.append(DebugSearchResult(
                chunk_id=0,
                content=doc.page_content,
                score=normalized_score,
                document_name=filename
            ))
            
        return DebugSearchResponse(
            results=output,
            query_analysis=getattr(results, "query_analysis", None),
            pipeline_trace=getattr(results, "pipeline_trace", None)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
