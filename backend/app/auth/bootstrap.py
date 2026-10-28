"""Create the first admin account from environment variables on startup.

Set ADMIN_EMAIL and ADMIN_PASSWORD (8+ characters) on the server. Nothing is
created if the account already exists; no default passwords are ever used.
"""

import logging
import os

from sqlmodel import Session, func, select

from app.auth.security import get_password_hash
from app.db import engine
from app.models.user import User, UserRole

logger = logging.getLogger(__name__)


def bootstrap_admin() -> None:
    email = (os.getenv("ADMIN_EMAIL") or "").strip().lower()
    password = os.getenv("ADMIN_PASSWORD") or ""
    if not email:
        return
    if len(password) < 8:
        logger.warning("ADMIN_EMAIL is set but ADMIN_PASSWORD is missing or shorter than 8 characters; skipping.")
        return
    with Session(engine) as session:
        if session.exec(select(User).where(func.lower(User.email) == email)).first():
            return
        session.add(User(email=email, hashed_password=get_password_hash(password), role=UserRole.ADMIN))
        session.commit()
        logger.info("Created admin account %s from environment", email)


_KNOWN_DEMO_CREDENTIALS = (("admin@ragops.com", "admin123"), ("client@ragops.com", "client123"))


def warn_about_demo_accounts() -> None:
    """Older versions seeded accounts with publicly known passwords. Warn loudly
    if any still accept those passwords (delete them from the admin Users list)."""
    from app.auth.security import verify_password
    with Session(engine) as session:
        for email, password in _KNOWN_DEMO_CREDENTIALS:
            user = session.exec(select(User).where(func.lower(User.email) == email)).first()
            if user and verify_password(password, user.hashed_password):
                logger.warning("SECURITY: demo account %s still uses its public default password. "
                               "Delete it or change its password.", email)
