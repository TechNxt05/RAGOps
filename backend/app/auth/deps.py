from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlmodel import Session, select
from starlette.concurrency import run_in_threadpool

from app.auth.security import ALGORITHM, SECRET_KEY, oauth2_scheme
from app.core import llm_keys
from app.db import get_session
from app.models.user import User, UserRole
from app.models.workspace import Workspace


def _user_from_token(token: str, session: Session) -> Optional[User]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
    email = payload.get("sub")
    if not email:
        return None
    return session.exec(select(User).where(User.email == email)).first()


def _load(token: str, session: Session):
    user = _user_from_token(token, session)
    if user is None:
        return None, None
    workspace = session.get(Workspace, user.workspace_id) if user.workspace_id else None
    return user, llm_keys.keys_for_workspace(workspace)


async def get_current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)) -> User:
    """Async on purpose: the workspace's LLM key scope is set in the request's own
    context, so it covers the endpoint, its thread-pool work and any stream."""
    user, keys = await run_in_threadpool(_load, token, session)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    llm_keys.set_scope(keys)
    return user


def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    """An admin of their own workspace (never of anyone else's)."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges",
        )
    return current_user


def get_current_workspace(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)) -> Workspace:
    workspace = session.get(Workspace, current_user.workspace_id) if current_user.workspace_id else None
    if workspace is None:
        raise HTTPException(status_code=403, detail="Your account is not part of a workspace")
    return workspace


def require_not_demo(workspace: Workspace, action: str) -> None:
    if workspace.is_demo:
        raise HTTPException(status_code=403, detail=f"{action} is disabled in the public demo. Create your own workspace to try it.")


async def get_optional_current_user(
    token: Optional[str] = Depends(OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)),
    session: Session = Depends(get_session),
) -> Optional[User]:
    if not token:
        return None
    user, keys = await run_in_threadpool(_load, token, session)
    if user is not None:
        llm_keys.set_scope(keys)
    return user
