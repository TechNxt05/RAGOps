from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List
from app.db import get_session
from app.models.rag import Project
from app.models.user import User
from app.auth.deps import get_current_user, get_current_admin

# Admin routes for managing projects
router = APIRouter(prefix="/rag/projects", tags=["rag-projects"])

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
    from app.models.rag import RAGConfig, Document, Chunk
    from app.models.chat import ChatSession, Message

    # 1. Delete RAG Config
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
