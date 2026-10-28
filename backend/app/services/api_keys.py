"""Per-project API keys for MCP / programmatic access. Only a SHA-256 hash is
stored; the plaintext key is shown once at creation."""

import hashlib
import hmac
import secrets
from datetime import datetime
from typing import Optional, Tuple

from sqlmodel import Session, select

from app.models.rag import ProjectApiKey

PREFIX = "rgo_"


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def create_key(session: Session, project_id: int, name: str, created_by: Optional[int]) -> Tuple[ProjectApiKey, str]:
    plaintext = PREFIX + secrets.token_urlsafe(32)
    record = ProjectApiKey(
        project_id=project_id, name=name, key_prefix=plaintext[:12],
        key_hash=_hash(plaintext), created_by=created_by,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record, plaintext


def resolve_key(session: Session, plaintext: str) -> Optional[ProjectApiKey]:
    if not plaintext or not plaintext.startswith(PREFIX):
        return None
    digest = _hash(plaintext)
    for record in session.exec(
        select(ProjectApiKey).where(ProjectApiKey.key_prefix == plaintext[:12]).where(ProjectApiKey.revoked == False)  # noqa: E712
    ).all():
        if hmac.compare_digest(record.key_hash, digest):
            record.last_used_at = datetime.utcnow()
            session.add(record)
            session.commit()
            return record
    return None
