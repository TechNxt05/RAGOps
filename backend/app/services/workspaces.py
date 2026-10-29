"""Workspace lifecycle: creation, invites, storage limits, demo seeding/reset,
and the cascading deletes that keep a workspace's data self-contained."""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional, Tuple

from fastapi import HTTPException
from sqlalchemy import delete, func, update
from sqlmodel import Session, select

from app.auth.security import get_password_hash
from app.core import llm_keys
from app.core.config import settings
from app.models.rag import (
    Chunk, ChunkEmbedding, Document, ExternalDatabase, IngestionJob, Project, ProjectApiKey,
    ProjectMember, RAGConfig, SemanticCacheEntry,
)
from app.models.chat import ChatSession, Message
from app.models.query_log import QueryLog
from app.models.usage import TokenUsage
from app.models.user import User, UserRole
from app.models.workspace import Workspace, WorkspaceInvite

logger = logging.getLogger(__name__)

DEMO_ADMIN_EMAIL = "demo-admin@ragops.demo"
DEMO_CLIENT_EMAIL = "demo-client@ragops.demo"


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def find_user(session: Session, email: str) -> Optional[User]:
    return session.exec(select(User).where(func.lower(User.email) == normalize_email(email))).first()


def create_workspace(
    session: Session,
    name: str,
    owner: User,
    keys: Optional[Dict[str, str]] = None,
    uses_platform_keys: bool = False,
    is_demo: bool = False,
) -> Workspace:
    workspace = Workspace(
        name=(name or "").strip()[:200] or f"{owner.email.split('@')[0]}'s workspace",
        is_demo=is_demo,
        uses_platform_keys=uses_platform_keys,
        llm_keys_encrypted=llm_keys.encrypt_keys(keys or {}),
    )
    session.add(workspace)
    session.flush()
    owner.workspace_id = workspace.id
    owner.role = UserRole.ADMIN
    session.add(owner)
    session.flush()
    workspace.owner_id = owner.id
    session.add(workspace)
    return workspace


INVITE_TTL_DAYS = 14


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_invite_token(invite: WorkspaceInvite) -> str:
    """(Re)issue the invite's secret; returns the plaintext to put in the link."""
    token = secrets.token_urlsafe(24)
    invite.token_hash = _token_hash(token)
    invite.created_at = datetime.utcnow()
    return token


def invite_by_token(session: Session, token: str) -> Optional[WorkspaceInvite]:
    if not token:
        return None
    invite = session.exec(select(WorkspaceInvite).where(WorkspaceInvite.token_hash == _token_hash(token))).first()
    if invite is None or invite.created_at < datetime.utcnow() - timedelta(days=INVITE_TTL_DAYS):
        return None
    return invite


def accept_invite(session: Session, user: User, invite: WorkspaceInvite) -> None:
    """Join the invited workspace and get the projects the admin picked."""
    user.workspace_id = invite.workspace_id
    user.role = UserRole.ADMIN if invite.role == "admin" else UserRole.CLIENT
    session.add(user)
    session.flush()
    for project_id in invite.project_ids or []:
        project = session.get(Project, project_id)
        if project and project.workspace_id == invite.workspace_id:
            session.add(ProjectMember(project_id=project_id, user_id=user.id, role="viewer"))
    for other in session.exec(
        select(WorkspaceInvite).where(func.lower(WorkspaceInvite.email) == normalize_email(user.email))
    ).all():
        session.delete(other)


# --- Limits --------------------------------------------------------------------

def _limited(workspace: Workspace) -> bool:
    return not workspace.uses_platform_keys or workspace.is_demo


def check_project_limit(session: Session, workspace: Workspace) -> None:
    limit = settings.workspace_max_projects
    if not limit or not _limited(workspace):
        return
    count = session.exec(select(func.count(Project.id)).where(Project.workspace_id == workspace.id)).one()
    if count >= limit:
        raise HTTPException(status_code=403, detail=f"This workspace has reached its limit of {limit} projects.")


def check_document_limit(session: Session, workspace: Optional[Workspace], adding: int) -> None:
    limit = settings.workspace_max_documents
    if workspace is None or not limit or not _limited(workspace):
        return
    count = session.exec(
        select(func.count(Document.id)).join(Project, Project.id == Document.project_id)
        .where(Project.workspace_id == workspace.id).where(Document.is_active == True)  # noqa: E712
    ).one()
    if count + adding > limit:
        raise HTTPException(
            status_code=403,
            detail=f"This workspace can hold at most {limit} documents ({count} already uploaded).",
        )


def check_member_limit(session: Session, workspace: Workspace) -> None:
    limit = settings.workspace_max_members
    if not limit or not _limited(workspace):
        return
    members = session.exec(select(func.count(User.id)).where(User.workspace_id == workspace.id)).one()
    invites = session.exec(select(func.count(WorkspaceInvite.id)).where(WorkspaceInvite.workspace_id == workspace.id)).one()
    if members + invites >= limit:
        raise HTTPException(status_code=403, detail=f"This workspace has reached its limit of {limit} members.")


# --- Deletes -------------------------------------------------------------------

def purge_project(session: Session, project_id: int) -> None:
    """Delete a project and everything that belongs to it (no commit)."""
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
    project = session.get(Project, project_id)
    if project:
        session.delete(project)


def delete_user(session: Session, user_id: int) -> None:
    """Delete an account but keep its history (logs, documents, keys) detached (no commit)."""
    session.exec(update(QueryLog).where(QueryLog.user_id == user_id).values(user_id=None))
    session.exec(update(Document).where(Document.uploaded_by == user_id).values(uploaded_by=None))
    session.exec(update(Project).where(Project.owner_id == user_id).values(owner_id=None))
    session.exec(update(ProjectApiKey).where(ProjectApiKey.created_by == user_id).values(created_by=None))
    session.exec(delete(ProjectMember).where(ProjectMember.user_id == user_id))
    user = session.get(User, user_id)
    if user:
        session.delete(user)


# --- Platform owner & demo -------------------------------------------------------

def ensure_own_workspace(session: Session, user: User, uses_platform_keys: bool) -> Workspace:
    if user.workspace_id:
        workspace = session.get(Workspace, user.workspace_id)
        if workspace:
            return workspace
    return create_workspace(session, "Main workspace", user, uses_platform_keys=uses_platform_keys)


def _demo_user(session: Session, email: str, workspace_id: int, role: UserRole) -> User:
    user = find_user(session, email)
    if user is None:
        # Nobody can log in with a password; the demo endpoint issues tokens.
        user = User(email=email, hashed_password=get_password_hash(secrets.token_urlsafe(32)), role=role)
    user.workspace_id, user.role = workspace_id, role
    session.add(user)
    session.flush()
    return user


def ensure_demo_workspace(session: Session) -> Tuple[Workspace, User, User]:
    workspace = session.exec(select(Workspace).where(Workspace.is_demo == True)).first()  # noqa: E712
    if workspace is None:
        workspace = Workspace(name="Demo workspace", is_demo=True, uses_platform_keys=True)
        session.add(workspace)
        session.flush()
    admin = _demo_user(session, DEMO_ADMIN_EMAIL, workspace.id, UserRole.ADMIN)
    client = _demo_user(session, DEMO_CLIENT_EMAIL, workspace.id, UserRole.CLIENT)
    workspace.owner_id = admin.id
    session.add(workspace)
    session.commit()
    return workspace, admin, client


def reset_expired_demo_projects(session: Session, workspace: Workspace) -> int:
    cutoff = datetime.utcnow() - timedelta(hours=settings.demo_reset_hours)
    expired: Iterable[int] = session.exec(
        select(Project.id).where(Project.workspace_id == workspace.id).where(Project.created_at < cutoff)
    ).all()
    expired = list(expired)
    for project_id in expired:
        purge_project(session, project_id)
    if expired:
        session.commit()
        logger.info("Demo reset: deleted %d expired demo project(s)", len(expired))
    return len(expired)
