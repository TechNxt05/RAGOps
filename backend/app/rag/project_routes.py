from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, delete, select
from typing import List, Optional
from pydantic import BaseModel, Field
from app.auth.access import readable_project_ids, require_project
from app.db import get_session
from app.jobs import queue
from app.models.rag import (
    Project, RAGConfig, Document, Chunk, ChunkEmbedding, ProjectMember, ProjectApiKey,
    ExternalDatabase, IngestionJob, SemanticCacheEntry,
)
from app.models.chat import ChatSession, Message
from app.models.query_log import QueryLog
from app.models.usage import TokenUsage
from app.models.user import User
from app.auth.deps import get_current_user, get_current_admin
from app.services.model_registry import normalize_model_name

# Admin routes for managing projects
router = APIRouter(prefix="/rag/projects", tags=["rag-projects"])


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None
    visibility: str = Field(default="all_users", pattern="^(all_users|members_only)$")


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None
    visibility: Optional[str] = Field(default=None, pattern="^(all_users|members_only)$")


class MemberIn(BaseModel):
    email: str
    role: str = Field(default="viewer", pattern="^(viewer|editor)$")


class ProjectRAGConfigPatch(BaseModel):
    primary_models: Optional[List[dict]] = None
    fallback_models: Optional[List[dict]] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = Field(default=None, ge=64, le=8192)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=4096)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    max_output_tokens: Optional[int] = Field(default=None, ge=16, le=32768)
    response_style: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    similarity_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    max_context_tokens: Optional[int] = Field(default=None, ge=256, le=200000)
    answer_only_from_docs: Optional[bool] = None
    hallucination_guard: Optional[bool] = None
    confidence_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    max_tokens: Optional[int] = None
    use_hybrid_search: Optional[bool] = None
    semantic_weight: Optional[float] = Field(default=None, ge=0, le=1)
    use_multi_query: Optional[bool] = None
    enable_self_correction: Optional[bool] = None
    enable_adaptive_router: Optional[bool] = None
    enable_query_rewrite: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    enable_reranker: Optional[bool] = None
    enable_cross_reference: Optional[bool] = None
    enable_conflict_detection: Optional[bool] = None
    enable_compression: Optional[bool] = None
    enable_claim_verification: Optional[bool] = None
    enable_reflective_generation: Optional[bool] = None
    enable_semantic_cache: Optional[bool] = None
    semantic_cache_threshold: Optional[float] = Field(default=None, ge=0.8, le=1)
    enable_cost_routing: Optional[bool] = None


def _active_config(session: Session, project_id: int) -> Optional[RAGConfig]:
    return session.exec(
        select(RAGConfig)
        .where(RAGConfig.project_id == project_id)
        .where(RAGConfig.is_active == True)  # noqa: E712
        .order_by(RAGConfig.created_at.desc())
    ).first()


def queue_reindex_if_embedding_changed(session: Session, project_id: int, old_model: Optional[str], new_model: Optional[str]) -> bool:
    if normalize_model_name(old_model) == normalize_model_name(new_model):
        return False
    queue.enqueue(session, "reindex_project", project_id=project_id, commit=False)
    return True


@router.patch("/{project_id}/config", response_model=RAGConfig)
def patch_project_rag_config(
    project_id: int,
    patch: ProjectRAGConfigPatch,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_admin),
):
    require_project(session, current_user, project_id, write=True)
    config = _active_config(session, project_id)
    if not config:
        config = RAGConfig(project_id=project_id)
        session.add(config)
        session.commit()
        session.refresh(config)
    old_embedding = config.embedding_model
    data = patch.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(config, key, value)
    if config.chunk_overlap >= config.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size")
    session.add(config)
    queue_reindex_if_embedding_changed(session, project_id, old_embedding, config.embedding_model)
    session.commit()
    session.refresh(config)
    return config


class ProjectPipelinePatch(BaseModel):
    pipeline_variant: str

@router.patch("/{project_id}/pipeline", response_model=RAGConfig)
def patch_project_pipeline(
    project_id: int,
    patch: ProjectPipelinePatch,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_admin),
):
    from app.benchmark.variants import VARIANT_CONFIGS, PipelineVariant

    require_project(session, current_user, project_id, write=True)
    variant_enum = next((v for v in PipelineVariant if v.value == patch.pipeline_variant), None)
    if not variant_enum or variant_enum not in VARIANT_CONFIGS:
        raise HTTPException(status_code=400, detail="Invalid pipeline variant")

    config = _active_config(session, project_id) or RAGConfig(project_id=project_id)
    config.pipeline_variant = patch.pipeline_variant

    # Variant presets use benchmark key names; map them onto RAGConfig toggles.
    aliases = {"use_self_correction": "enable_self_correction", "use_reranker": "enable_reranker"}
    for k, v in VARIANT_CONFIGS[variant_enum].items():
        k = aliases.get(k, k)
        if hasattr(config, k):
            setattr(config, k, v)

    session.add(config)
    session.commit()
    session.refresh(config)
    return config


@router.post("/", response_model=Project)
def create_project(body: ProjectCreate, session: Session = Depends(get_session), current_user: User = Depends(get_current_admin)):
    project = Project(name=body.name, description=body.description, visibility=body.visibility, owner_id=current_user.id)
    session.add(project)
    session.commit()
    session.refresh(project)
    session.add(RAGConfig(project_id=project.id))
    session.commit()
    session.refresh(project)
    return project


@router.get("/", response_model=List[Project])
def list_projects(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    ids = readable_project_ids(session, current_user)
    query = select(Project).order_by(Project.created_at.desc())
    if ids is not None:
        query = query.where(Project.id.in_(ids))
    return session.exec(query).all()


@router.get("/{project_id}", response_model=Project)
def get_project(project_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return require_project(session, current_user, project_id)


@router.patch("/{project_id}", response_model=Project)
def update_project(project_id: int, body: ProjectUpdate, session: Session = Depends(get_session), current_user: User = Depends(get_current_admin)):
    project = require_project(session, current_user, project_id, write=True)
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(project, key, value)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


# --- Members --------------------------------------------------------------------

@router.get("/{project_id}/members")
def list_members(project_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_admin)):
    require_project(session, current_user, project_id)
    rows = session.exec(
        select(ProjectMember, User).join(User, User.id == ProjectMember.user_id).where(ProjectMember.project_id == project_id)
    ).all()
    return [{"user_id": u.id, "email": u.email, "role": m.role, "added_at": m.created_at} for m, u in rows]


@router.post("/{project_id}/members")
def add_member(project_id: int, body: MemberIn, session: Session = Depends(get_session), current_user: User = Depends(get_current_admin)):
    require_project(session, current_user, project_id)
    user = session.exec(select(User).where(User.email == body.email.strip().lower())).first() \
        or session.exec(select(User).where(User.email == body.email.strip())).first()
    if not user:
        raise HTTPException(status_code=404, detail="No user with that email. Create the account first.")
    member = session.exec(
        select(ProjectMember).where(ProjectMember.project_id == project_id).where(ProjectMember.user_id == user.id)
    ).first()
    if member:
        member.role = body.role
    else:
        member = ProjectMember(project_id=project_id, user_id=user.id, role=body.role)
    session.add(member)
    session.commit()
    return {"user_id": user.id, "email": user.email, "role": member.role}


@router.delete("/{project_id}/members/{user_id}")
def remove_member(project_id: int, user_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_admin)):
    require_project(session, current_user, project_id)
    session.exec(delete(ProjectMember).where(ProjectMember.project_id == project_id).where(ProjectMember.user_id == user_id))
    session.commit()
    return {"ok": True}


@router.delete("/{project_id}")
def delete_project(project_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_admin)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    doc_ids = list(session.exec(select(Document.id).where(Document.project_id == project_id)).all())
    session_ids = list(session.exec(select(ChatSession.id).where(ChatSession.project_id == project_id)).all())

    session.exec(delete(ChunkEmbedding).where(ChunkEmbedding.project_id == project_id))
    if doc_ids:
        session.exec(delete(Chunk).where(Chunk.document_id.in_(doc_ids)))
    session.exec(delete(QueryLog).where(QueryLog.project_id == project_id))
    if session_ids:
        session.exec(delete(Message).where(Message.session_id.in_(session_ids)))
        session.exec(delete(ChatSession).where(ChatSession.id.in_(session_ids)))
    session.exec(delete(Document).where(Document.project_id == project_id))
    for model in (RAGConfig, ProjectMember, ProjectApiKey, ExternalDatabase, IngestionJob, SemanticCacheEntry, TokenUsage):
        session.exec(delete(model).where(model.project_id == project_id))
    session.delete(project)
    session.commit()
    return {"ok": True}
