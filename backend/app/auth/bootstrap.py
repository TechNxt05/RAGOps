"""Startup bootstrap: the platform owner's admin account and the demo workspace.

Set ADMIN_EMAIL and ADMIN_PASSWORD (8+ characters) on the server. The account is
created once, as admin of a workspace that uses the server's own LLM keys. No
default passwords are ever used.
"""

import logging
import os

from sqlmodel import Session

from app.auth.security import get_password_hash
from app.core.config import settings
from app.db import engine
from app.models.user import User, UserRole
from app.services import workspaces

logger = logging.getLogger(__name__)


def bootstrap_admin() -> None:
    email = workspaces.normalize_email(os.getenv("ADMIN_EMAIL") or "")
    password = os.getenv("ADMIN_PASSWORD") or ""
    if not email:
        return
    with Session(engine) as session:
        user = workspaces.find_user(session, email)
        if user is None:
            if len(password) < 8:
                logger.warning("ADMIN_EMAIL is set but ADMIN_PASSWORD is missing or shorter than 8 characters; skipping.")
                return
            user = User(email=email, hashed_password=get_password_hash(password), role=UserRole.ADMIN)
            session.add(user)
            session.flush()
            logger.info("Created admin account %s from environment", email)
        # The platform owner's workspace runs on the server's own API keys.
        workspace = workspaces.ensure_own_workspace(session, user, uses_platform_keys=True)
        if not workspace.uses_platform_keys and workspace.owner_id == user.id:
            workspace.uses_platform_keys = True
            session.add(workspace)
        session.commit()


def bootstrap_demo() -> None:
    if not settings.demo_mode:
        return
    with Session(engine) as session:
        workspaces.ensure_demo_workspace(session)


# Older versions seeded these with publicly known passwords. Logins with them
# are refused (see auth.routes); the accounts themselves are left untouched.
KNOWN_DEMO_CREDENTIALS = (("admin@ragops.com", "admin123"), ("client@ragops.com", "client123"))


def is_retired_demo_login(email: str, password: str) -> bool:
    return (workspaces.normalize_email(email), password) in KNOWN_DEMO_CREDENTIALS
