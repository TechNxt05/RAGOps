from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List, Optional
from pydantic import BaseModel
from app.db import get_session
from app.models.rag import Project, RAGConfig, Document, Chunk
from app.models.chat import ChatSession, Message
from app.models.query_log import QueryLog
from app.models.user import User
from app.auth.deps import get_current_user, get_current_admin

# Admin routes for managing projects
router = APIRouter(prefix="/rag/projects", tags=["rag-projects"])


class ProjectRAGConfigPatch(BaseModel):
    primary_llm_provider: Optional[str] = None
    primary_llm_name: Optional[str] = None
    fallback_llm_provider: Optional[str] = None
    fallback_llm_name: Optional[str] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    response_style: Optional[str] = None
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    max_context_tokens: Optional[int] = None
    answer_only_from_docs: Optional[bool] = None
    hallucination_guard: Optional[bool] = None
    max_tokens: Optional[int] = None


@router.patch("/{project_id}/config", response_model=RAGConfig)
def patch_project_rag_config(
    project_id: int,
    patch: ProjectRAGConfigPatch,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_admin),
):
    config = session.exec(
        select(RAGConfig)
        .where(RAGConfig.project_id == project_id)
        .where(RAGConfig.is_active == True)
        .order_by(RAGConfig.created_at.desc())
    ).first()
    if not config:
        config = RAGConfig(project_id=project_id)
        session.add(config)
        session.commit()
        session.refresh(config)
    data = patch.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(config, key, value)
    session.add(config)
    session.commit()
    session.refresh(config)
    return config


@router.post("/", response_model=Project)
def create_project(project: Project, session: Session = Depends(get_session), current_user: User = Depends(get_current_admin)):
    session.add(project)
    session.commit()
    session.refresh(project)
    return project

@router.get("/", response_model=List[Project])
def list_projects(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    # Both Admin and Client can see projects
    projects = session.exec(select(Project)).all()
    return projects

@router.get("/{project_id}", response_model=Project)
def get_project(project_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.delete("/{project_id}")
def delete_project(project_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_admin)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Manual Cascade Delete to handle Foreign Key Constraints
    logs = session.exec(select(QueryLog).where(QueryLog.project_id == project_id)).all()
    for lg in logs:
        session.delete(lg)
    configs = session.exec(select(RAGConfig).where(RAGConfig.project_id == project_id)).all()
    for c in configs:
        session.delete(c)
    
    # 2. Delete Documents & Chunks
    docs = session.exec(select(Document).where(Document.project_id == project_id)).all()
    for d in docs:
        # Delete chunks for each doc
        chunks = session.exec(select(Chunk).where(Chunk.document_id == d.id)).all()
        for ch in chunks:
            session.delete(ch)
        session.delete(d)
        
    # 3. Delete Chat Sessions & Messages
    chats = session.exec(select(ChatSession).where(ChatSession.project_id == project_id)).all()
    for c in chats:
        msgs = session.exec(select(Message).where(Message.session_id == c.id)).all()
        for m in msgs:
            session.delete(m)
        session.delete(c)

    session.delete(project)
    session.commit()
    return {"ok": True}
