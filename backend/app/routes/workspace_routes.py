"""Workspace settings for its admins: name, LLM API keys, invites, and which
projects each member can open."""

from typing import Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, delete, select
from starlette.concurrency import run_in_threadpool

from app.auth.deps import get_current_admin, get_current_workspace, require_not_demo
from app.auth.routes import invite_email
from app.core import llm_keys
from app.core.config import settings
from app.db import get_session
from app.models.rag import Project, ProjectMember
from app.models.user import User
from app.models.workspace import Workspace, WorkspaceInvite
from app.services import workspaces

router = APIRouter(prefix="/workspace", tags=["workspace"])


class WorkspacePatch(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class KeysUpdate(BaseModel):
    # Omitted: keep. Empty string: remove. Anything else: replace.
    groq: Optional[str] = Field(default=None, max_length=500)
    google: Optional[str] = Field(default=None, max_length=500)
    openrouter: Optional[str] = Field(default=None, max_length=500)
    verify: bool = True


class InviteIn(BaseModel):
    email: str = Field(max_length=320)
    role: str = Field(default="client", pattern="^(client|admin)$")
    project_ids: List[int] = Field(default_factory=list, max_length=200)


class MemberProjects(BaseModel):
    project_ids: List[int] = Field(default_factory=list, max_length=500)


def _summary(session: Session, workspace: Workspace) -> dict:
    keys = llm_keys.decrypt_keys(workspace.llm_keys_encrypted)
    projects = session.exec(select(Project).where(Project.workspace_id == workspace.id).order_by(Project.name)).all()
    project_ids = [p.id for p in projects]
    memberships: Dict[int, List[int]] = {}
    if project_ids:
        for m in session.exec(select(ProjectMember).where(ProjectMember.project_id.in_(project_ids))).all():
            memberships.setdefault(m.user_id, []).append(m.project_id)
    members = session.exec(select(User).where(User.workspace_id == workspace.id).order_by(User.id)).all()
    invites = session.exec(
        select(WorkspaceInvite).where(WorkspaceInvite.workspace_id == workspace.id).order_by(WorkspaceInvite.created_at.desc())
    ).all()
    return {
        "id": workspace.id,
        "name": workspace.name,
        "is_demo": workspace.is_demo,
        "uses_platform_keys": workspace.uses_platform_keys,
        "llm_keys": {p: (llm_keys.mask(keys[p]) if p in keys else None) for p in llm_keys.WORKSPACE_PROVIDERS},
        "limits": {
            "projects": settings.workspace_max_projects, "documents": settings.workspace_max_documents,
            "members": settings.workspace_max_members,
        } if (not workspace.uses_platform_keys or workspace.is_demo) else None,
        "projects": [{"id": p.id, "name": p.name, "visibility": p.visibility} for p in projects],
        "members": [
            {"id": u.id, "email": u.email, "role": u.role, "is_owner": u.id == workspace.owner_id,
             "project_ids": sorted(memberships.get(u.id, []))}
            for u in members
        ],
        "invites": [
            {"id": i.id, "email": i.email, "role": i.role, "project_ids": i.project_ids or [],
             "created_at": i.created_at, "expires_in_days": max(0, workspaces.INVITE_TTL_DAYS - (_age_days(i)))}
            for i in invites
        ],
    }


def _age_days(invite: WorkspaceInvite) -> int:
    from datetime import datetime
    return (datetime.utcnow() - invite.created_at).days


@router.get("")
def get_workspace(
    session: Session = Depends(get_session),
    _: User = Depends(get_current_admin),
    workspace: Workspace = Depends(get_current_workspace),
):
    return _summary(session, workspace)


@router.patch("")
def rename_workspace(
    body: WorkspacePatch,
    session: Session = Depends(get_session),
    _: User = Depends(get_current_admin),
    workspace: Workspace = Depends(get_current_workspace),
):
    require_not_demo(workspace, "Renaming the workspace")
    workspace.name = body.name.strip()
    session.add(workspace)
    session.commit()
    return _summary(session, workspace)


# --- LLM keys ----------------------------------------------------------------------

def _verify_key(provider: str, key: str) -> Optional[str]:
    """None if the provider accepts the key, else a short reason."""
    try:
        if provider == "groq":
            r = httpx.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
        elif provider == "google":
            r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models", params={"key": key, "pageSize": 1}, timeout=10)
        else:
            r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=10)
    except httpx.HTTPError:
        return None  # provider unreachable: don't block saving on our network
    if r.status_code in (400, 401, 403):
        return "the provider rejected this key"
    return None


@router.put("/keys")
async def update_keys(
    body: KeysUpdate,
    session: Session = Depends(get_session),
    _: User = Depends(get_current_admin),
    workspace: Workspace = Depends(get_current_workspace),
):
    require_not_demo(workspace, "Changing API keys")
    if workspace.uses_platform_keys:
        raise HTTPException(status_code=400, detail="This workspace uses the server's own API keys (set in the hosting environment).")
    keys = llm_keys.decrypt_keys(workspace.llm_keys_encrypted)
    changes = {p: getattr(body, p) for p in llm_keys.WORKSPACE_PROVIDERS if getattr(body, p) is not None}
    if body.verify:
        for provider, value in changes.items():
            if value.strip():
                reason = await run_in_threadpool(_verify_key, provider, value.strip())
                if reason:
                    raise HTTPException(status_code=400, detail=f"{provider.title()} key not saved: {reason}.")
    for provider, value in changes.items():
        if value.strip():
            keys[provider] = value.strip()
        else:
            keys.pop(provider, None)
    workspace.llm_keys_encrypted = llm_keys.encrypt_keys(keys)
    session.add(workspace)
    session.commit()
    return _summary(session, workspace)


# --- Invites -------------------------------------------------------------------------

@router.post("/invites")
def create_invite(
    body: InviteIn,
    session: Session = Depends(get_session),
    admin: User = Depends(get_current_admin),
    workspace: Workspace = Depends(get_current_workspace),
):
    require_not_demo(workspace, "Inviting people")
    invite, token = invite_email(session, workspace, body.email, body.role, body.project_ids, admin.id)
    session.commit()
    return {"id": invite.id, "email": invite.email, "role": invite.role, "token": token}


def _own_invite(session: Session, invite_id: int, workspace: Workspace) -> WorkspaceInvite:
    invite = session.get(WorkspaceInvite, invite_id)
    if not invite or invite.workspace_id != workspace.id:
        raise HTTPException(status_code=404, detail="Invite not found")
    return invite


@router.post("/invites/{invite_id}/link")
def regenerate_invite_link(
    invite_id: int,
    session: Session = Depends(get_session),
    _: User = Depends(get_current_admin),
    workspace: Workspace = Depends(get_current_workspace),
):
    """Issue a fresh link (the old one stops working)."""
    invite = _own_invite(session, invite_id, workspace)
    token = workspaces.new_invite_token(invite)
    session.add(invite)
    session.commit()
    return {"id": invite.id, "email": invite.email, "role": invite.role, "token": token}


@router.delete("/invites/{invite_id}")
def delete_invite(
    invite_id: int,
    session: Session = Depends(get_session),
    _: User = Depends(get_current_admin),
    workspace: Workspace = Depends(get_current_workspace),
):
    session.delete(_own_invite(session, invite_id, workspace))
    session.commit()
    return {"ok": True}


# --- Project access per member -------------------------------------------------------

@router.put("/members/{user_id}/projects")
def set_member_projects(
    user_id: int,
    body: MemberProjects,
    session: Session = Depends(get_session),
    _: User = Depends(get_current_admin),
    workspace: Workspace = Depends(get_current_workspace),
):
    """Replace the set of projects this member is assigned to. Projects set to
    'members only' are then visible to exactly the assigned members."""
    member = session.get(User, user_id)
    if not member or member.workspace_id != workspace.id:
        raise HTTPException(status_code=404, detail="User not found")
    own_ids = set(session.exec(select(Project.id).where(Project.workspace_id == workspace.id)).all())
    wanted = set(body.project_ids)
    if not wanted <= own_ids:
        raise HTTPException(status_code=404, detail="Project not found")
    existing = {
        m.project_id: m for m in session.exec(select(ProjectMember).where(ProjectMember.user_id == user_id)).all()
        if m.project_id in own_ids
    }
    removed = set(existing) - wanted
    if removed:
        session.exec(delete(ProjectMember).where(ProjectMember.user_id == user_id).where(ProjectMember.project_id.in_(removed)))
    for project_id in wanted - set(existing):
        session.add(ProjectMember(project_id=project_id, user_id=user_id, role="viewer"))
    session.commit()
    return _summary(session, workspace)
