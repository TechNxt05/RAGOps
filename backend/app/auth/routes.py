from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, func, select
from app.db import get_session
from app.models.user import User, UserRole
from app.models.workspace import Workspace
from app.auth.bootstrap import is_retired_demo_login
from app.auth.security import verify_password, get_password_hash, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from app.auth.deps import get_current_user, get_current_admin, get_current_workspace, require_not_demo
from app.core.config import settings
from app.core.rate_limit import rate_limiter
from app.services import workspaces
from datetime import timedelta
from pydantic import BaseModel, Field

router = APIRouter(prefix="/auth", tags=["auth"])

class UserCreate(BaseModel):
    email: str
    password: str = Field(min_length=8)
    # From an admin's invite link: join that workspace instead of creating one.
    invite_token: Optional[str] = Field(default=None, max_length=200)
    # Creating a new workspace:
    workspace_name: Optional[str] = Field(default=None, max_length=200)
    llm_keys: Optional[Dict[str, str]] = None
    client_emails: List[str] = Field(default_factory=list, max_length=100)
    # Accepted for backward compatibility with older clients but ignored.
    role: Optional[str] = None

class AdminUserCreate(BaseModel):
    email: str
    password: str = Field(min_length=8)
    role: UserRole = UserRole.CLIENT

class UserRoleUpdate(BaseModel):
    role: UserRole

class WorkspaceBrief(BaseModel):
    id: int
    name: str
    is_demo: bool

class UserPublic(BaseModel):
    id: int
    email: str
    role: UserRole

class Me(UserPublic):
    workspace: Optional[WorkspaceBrief] = None

class InviteLink(BaseModel):
    email: str
    token: str

class RegisterResult(UserPublic):
    joined_existing_workspace: bool = False
    workspace_name: Optional[str] = None
    # Invites created for client_emails: the frontend turns tokens into links.
    invites: List[InviteLink] = Field(default_factory=list)

class Token(BaseModel):
    access_token: str
    token_type: str

class DemoLogin(BaseModel):
    role: str = Field(pattern="^(admin|client)$")


def _create_user(session: Session, email: str, password: str, role: UserRole, workspace_id: Optional[int]) -> User:
    email = workspaces.normalize_email(email)
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    if workspaces.find_user(session, email):
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=email, hashed_password=get_password_hash(password), role=role, workspace_id=workspace_id)
    session.add(user)
    session.flush()
    return user


def _token_for(user: User) -> Dict[str, str]:
    token = create_access_token(subject=user.email, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}


@router.post("/register", response_model=RegisterResult)
def register(user_in: UserCreate, request: Request, session: Session = Depends(get_session)):
    """With an invite link: join the inviting workspace. Otherwise: create a new
    workspace and become its admin (when self-registration is enabled)."""
    rate_limiter.check(f"register:{request.client.host if request.client else '-'}", 10)
    if user_in.invite_token:
        invite = workspaces.invite_by_token(session, user_in.invite_token)
        if invite is None:
            raise HTTPException(status_code=400, detail="This invite link is invalid or has expired. Ask your admin for a new one.")
        if workspaces.normalize_email(user_in.email) != invite.email:
            raise HTTPException(status_code=400, detail=f"This invite is for {invite.email}. Register with that email.")
        user = _create_user(session, user_in.email, user_in.password, UserRole.CLIENT, None)
        workspaces.accept_invite(session, user, invite)
        session.commit()
        workspace = session.get(Workspace, user.workspace_id)
        return RegisterResult(id=user.id, email=user.email, role=user.role,
                              joined_existing_workspace=True, workspace_name=workspace.name if workspace else None)

    first_account = session.exec(select(func.count(User.id))).one() == 0
    if not settings.allow_self_registration and not first_account:
        raise HTTPException(status_code=403, detail="Self-registration is disabled. Ask your admin for an invite link.")
    user = _create_user(session, user_in.email, user_in.password, UserRole.ADMIN, None)
    # The very first account on a fresh install is the operator: their workspace
    # runs on the server's own keys. Everyone after brings their own.
    workspace = workspaces.create_workspace(
        session, user_in.workspace_name or "", user, keys=user_in.llm_keys,
        uses_platform_keys=first_account and settings.first_user_is_admin,
    )
    links = []
    for email in dict.fromkeys(workspaces.normalize_email(e) for e in user_in.client_emails if e.strip()):
        if email == user.email:
            continue
        _, token = invite_email(session, workspace, email, "client", [], user.id)
        links.append(InviteLink(email=email, token=token))
    session.commit()
    return RegisterResult(id=user.id, email=user.email, role=user.role, workspace_name=workspace.name, invites=links)


@router.get("/invite/{token}")
def invite_info(token: str, session: Session = Depends(get_session)):
    """Prefill the join form from an invite link."""
    invite = workspaces.invite_by_token(session, token)
    if invite is None:
        raise HTTPException(status_code=404, detail="This invite link is invalid or has expired. Ask your admin for a new one.")
    workspace = session.get(Workspace, invite.workspace_id)
    return {"email": invite.email, "workspace_name": workspace.name if workspace else None, "role": invite.role}


@router.post("/token", response_model=Token)
def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)):
    rate_limiter.check(f"login:{request.client.host if request.client else '-'}", 20)
    user = workspaces.find_user(session, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if is_retired_demo_login(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This old demo password is public and no longer accepted. Use the demo buttons, or ask the admin to reset this account.",
        )
    return _token_for(user)


@router.get("/demo")
def demo_status():
    return {"enabled": settings.demo_mode, "reset_hours": settings.demo_reset_hours}


@router.post("/demo", response_model=Token)
def demo_login(body: DemoLogin, request: Request, session: Session = Depends(get_session)):
    """One-click login into the shared demo workspace."""
    if not settings.demo_mode:
        raise HTTPException(status_code=404, detail="The demo is not enabled on this server")
    rate_limiter.check(f"demo-login:{request.client.host if request.client else '-'}", 10)
    workspace, admin, client = workspaces.ensure_demo_workspace(session)
    workspaces.reset_expired_demo_projects(session, workspace)
    return _token_for(admin if body.role == "admin" else client)


@router.get("/me", response_model=Me)
def read_users_me(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    workspace = session.get(Workspace, current_user.workspace_id) if current_user.workspace_id else None
    return Me(
        id=current_user.id, email=current_user.email, role=current_user.role,
        workspace=WorkspaceBrief(id=workspace.id, name=workspace.name, is_demo=workspace.is_demo) if workspace else None,
    )


# --- Admin user management (own workspace only) ---------------------------------

def invite_email(session: Session, workspace: Workspace, email: str, role: str, project_ids: List[int], created_by: Optional[int]):
    """Create (or refresh) an invite. Returns (invite, plaintext token)."""
    from app.models.workspace import WorkspaceInvite
    from app.models.rag import Project
    email = workspaces.normalize_email(email)
    if "@" not in email:
        raise HTTPException(status_code=400, detail=f"'{email}' is not a valid email address")
    existing = workspaces.find_user(session, email)
    if existing is not None:
        if existing.workspace_id == workspace.id:
            raise HTTPException(status_code=400, detail=f"{email} is already in this workspace")
        raise HTTPException(status_code=409, detail=f"{email} already has an account in another workspace")
    valid = [pid for pid in project_ids if (p := session.get(Project, pid)) and p.workspace_id == workspace.id]
    invite = session.exec(
        select(WorkspaceInvite).where(WorkspaceInvite.workspace_id == workspace.id).where(WorkspaceInvite.email == email)
    ).first() or WorkspaceInvite(workspace_id=workspace.id, email=email)
    if invite.id is None:
        workspaces.check_member_limit(session, workspace)
    invite.role, invite.project_ids, invite.created_by = role, valid, created_by
    token = workspaces.new_invite_token(invite)
    session.add(invite)
    session.flush()
    return invite, token


def _member_or_404(session: Session, user_id: int, admin: User) -> User:
    user = session.get(User, user_id)
    if not user or user.workspace_id != admin.workspace_id:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/users", response_model=List[UserPublic])
def list_users(session: Session = Depends(get_session), admin: User = Depends(get_current_admin)):
    return session.exec(select(User).where(User.workspace_id == admin.workspace_id).order_by(User.id)).all()


@router.post("/users", response_model=UserPublic)
def admin_create_user(
    body: AdminUserCreate,
    session: Session = Depends(get_session),
    admin: User = Depends(get_current_admin),
    workspace: Workspace = Depends(get_current_workspace),
):
    require_not_demo(workspace, "Creating accounts")
    workspaces.check_member_limit(session, workspace)
    user = _create_user(session, body.email, body.password, body.role, workspace.id)
    session.commit()
    session.refresh(user)
    return user


@router.patch("/users/{user_id}", response_model=UserPublic)
def admin_update_role(
    user_id: int,
    body: UserRoleUpdate,
    session: Session = Depends(get_session),
    admin: User = Depends(get_current_admin),
    workspace: Workspace = Depends(get_current_workspace),
):
    require_not_demo(workspace, "Changing roles")
    user = _member_or_404(session, user_id, admin)
    if user.id == admin.id and body.role != UserRole.ADMIN:
        raise HTTPException(status_code=400, detail="You cannot remove your own admin role")
    user.role = body.role
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@router.delete("/users/{user_id}")
def admin_delete_user(
    user_id: int,
    session: Session = Depends(get_session),
    admin: User = Depends(get_current_admin),
    workspace: Workspace = Depends(get_current_workspace),
):
    require_not_demo(workspace, "Deleting accounts")
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    _member_or_404(session, user_id, admin)
    if user_id == workspace.owner_id:
        raise HTTPException(status_code=400, detail="The workspace owner cannot be deleted")
    workspaces.delete_user(session, user_id)
    session.commit()
    return {"ok": True}
