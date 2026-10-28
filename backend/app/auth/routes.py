from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select, func
from app.db import get_session
from app.models.user import User, UserRole
from app.auth.security import verify_password, get_password_hash, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from app.auth.deps import get_current_user, get_current_admin
from app.core.config import settings
from datetime import timedelta
from pydantic import BaseModel, Field

router = APIRouter(prefix="/auth", tags=["auth"])

class UserCreate(BaseModel):
    email: str
    password: str = Field(min_length=8)
    # Accepted for backward compatibility with older clients but ignored:
    # self-registration can never grant ADMIN. Admins use POST /auth/users.
    role: Optional[str] = None

class AdminUserCreate(BaseModel):
    email: str
    password: str = Field(min_length=8)
    role: UserRole = UserRole.CLIENT

class UserRoleUpdate(BaseModel):
    role: UserRole

class UserPublic(BaseModel):
    id: int
    email: str
    role: UserRole

class Token(BaseModel):
    access_token: str
    token_type: str


def _create_user(session: Session, email: str, password: str, role: UserRole) -> User:
    email = email.strip().lower()
    if session.exec(select(User).where(func.lower(User.email) == email)).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=email, hashed_password=get_password_hash(password), role=role)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@router.post("/register", response_model=UserPublic)
def register(user_in: UserCreate, session: Session = Depends(get_session)):
    has_users = session.exec(select(func.count(User.id))).one() > 0
    if not settings.allow_self_registration and has_users:
        raise HTTPException(status_code=403, detail="Self-registration is disabled. Ask an admin for an account.")
    # Bootstrap: the very first account on a fresh install becomes the admin.
    role = UserRole.ADMIN if (not has_users and settings.first_user_is_admin) else UserRole.CLIENT
    return _create_user(session, user_in.email, user_in.password, role)


@router.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)):
    email = form_data.username.strip().lower()
    user = session.exec(select(User).where(func.lower(User.email) == email)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user.email, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserPublic)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user


# --- Admin user management --------------------------------------------------

@router.get("/users", response_model=List[UserPublic])
def list_users(session: Session = Depends(get_session), _: User = Depends(get_current_admin)):
    return session.exec(select(User).order_by(User.id)).all()


@router.post("/users", response_model=UserPublic)
def admin_create_user(body: AdminUserCreate, session: Session = Depends(get_session), _: User = Depends(get_current_admin)):
    return _create_user(session, body.email, body.password, body.role)


@router.patch("/users/{user_id}", response_model=UserPublic)
def admin_update_role(
    user_id: int,
    body: UserRoleUpdate,
    session: Session = Depends(get_session),
    admin: User = Depends(get_current_admin),
):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id and body.role != UserRole.ADMIN:
        raise HTTPException(status_code=400, detail="You cannot remove your own admin role")
    user.role = body.role
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@router.delete("/users/{user_id}")
def admin_delete_user(user_id: int, session: Session = Depends(get_session), admin: User = Depends(get_current_admin)):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Keep history (logs, documents, keys) but detach it from the deleted account.
    from sqlalchemy import update
    from app.models.query_log import QueryLog
    from app.models.rag import Document, Project, ProjectApiKey, ProjectMember
    session.exec(update(QueryLog).where(QueryLog.user_id == user_id).values(user_id=None))
    session.exec(update(Document).where(Document.uploaded_by == user_id).values(uploaded_by=None))
    session.exec(update(Project).where(Project.owner_id == user_id).values(owner_id=None))
    session.exec(update(ProjectApiKey).where(ProjectApiKey.created_by == user_id).values(created_by=None))
    for m in session.exec(select(ProjectMember).where(ProjectMember.user_id == user_id)).all():
        session.delete(m)
    session.delete(user)
    session.commit()
    return {"ok": True}
