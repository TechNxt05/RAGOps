from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from app.db import get_session
from app.models.rag import RAGConfig
from app.models.user import User
from app.auth.deps import get_current_admin
from datetime import datetime

router = APIRouter(prefix="/rag/config", tags=["rag-config"])

from app.auth.deps import get_current_admin, get_current_user

@router.get("/", response_model=RAGConfig)
def get_rag_config(project_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    # Get the latest active config for the specific project
    config = session.exec(select(RAGConfig).where(RAGConfig.project_id == project_id).where(RAGConfig.is_active == True).order_by(RAGConfig.created_at.desc())).first()
    if not config:
        # Create default if none exists for this project
        config = RAGConfig(project_id=project_id)
        session.add(config)
        session.commit()
        session.refresh(config)
    return config

@router.post("/", response_model=RAGConfig)
def update_rag_config(config_in: RAGConfig, session: Session = Depends(get_session), current_user: User = Depends(get_current_admin)):
    if not config_in.project_id:
        raise HTTPException(status_code=400, detail="project_id is required")

    # Deactivate currently active configs for THIS project
    active_configs = session.exec(select(RAGConfig).where(RAGConfig.project_id == config_in.project_id).where(RAGConfig.is_active == True)).all()
    for conf in active_configs:
        conf.is_active = False
        session.add(conf)
    
    # Create new config
    # We use dict() exclude to safely copy all user-provided fields 
    # while ignoring system-managed fields
    config_data = config_in.dict(exclude={'id', 'created_at', 'is_active'})
    
    new_config = RAGConfig(
        **config_data,
        is_active=True,
        created_at=datetime.utcnow()
    )
    session.add(new_config)
    session.commit()
    session.refresh(new_config)
    
    return new_config
