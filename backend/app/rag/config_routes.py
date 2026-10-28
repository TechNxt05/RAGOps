from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.auth.access import require_project
from app.auth.deps import get_current_admin, get_current_user
from app.db import get_session
from app.models.rag import RAGConfig
from app.models.user import User
from app.rag.project_routes import queue_reindex_if_embedding_changed

router = APIRouter(prefix="/rag/config", tags=["rag-config"])


def _active(session: Session, project_id: int):
    return session.exec(
        select(RAGConfig)
        .where(RAGConfig.project_id == project_id)
        .where(RAGConfig.is_active == True)  # noqa: E712
        .order_by(RAGConfig.created_at.desc())
    ).first()


@router.get("/", response_model=RAGConfig)
def get_rag_config(project_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    require_project(session, current_user, project_id)
    config = _active(session, project_id)
    if not config:
        config = RAGConfig(project_id=project_id)
        session.add(config)
        session.commit()
        session.refresh(config)
    return config


@router.post("/", response_model=RAGConfig)
def update_rag_config(config_in: RAGConfig, session: Session = Depends(get_session), current_user: User = Depends(get_current_admin)):
    """Saves a new config version for the project (previous versions are kept
    inactive for audit/rollback)."""
    if not config_in.project_id:
        raise HTTPException(status_code=400, detail="project_id is required")
    require_project(session, current_user, config_in.project_id, write=True)
    if config_in.chunk_overlap >= config_in.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size")

    previous = _active(session, config_in.project_id)
    old_embedding = previous.embedding_model if previous else None
    for conf in session.exec(
        select(RAGConfig).where(RAGConfig.project_id == config_in.project_id).where(RAGConfig.is_active == True)  # noqa: E712
    ).all():
        conf.is_active = False
        session.add(conf)

    config_data = config_in.model_dump(exclude={"id", "created_at", "is_active"})
    new_config = RAGConfig(**config_data, is_active=True, created_at=datetime.utcnow())
    session.add(new_config)
    if previous is not None:
        queue_reindex_if_embedding_changed(session, config_in.project_id, old_embedding, new_config.embedding_model)
    session.commit()
    session.refresh(new_config)
    return new_config
