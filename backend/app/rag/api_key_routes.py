from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.auth.access import require_project
from app.auth.deps import get_current_user
from app.db import get_session
from app.models.rag import ProjectApiKey
from app.models.user import User
from app.services.api_keys import create_key

router = APIRouter(prefix="/rag/projects/{project_id}/api-keys", tags=["api-keys"])


class KeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)


@router.get("")
def list_keys(project_id: int, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    require_project(session, user, project_id, write=True)
    keys: List[ProjectApiKey] = session.exec(
        select(ProjectApiKey).where(ProjectApiKey.project_id == project_id).order_by(ProjectApiKey.id.desc())
    ).all()
    return [
        {"id": k.id, "name": k.name, "prefix": k.key_prefix, "created_at": k.created_at,
         "last_used_at": k.last_used_at, "revoked": k.revoked}
        for k in keys
    ]


@router.post("")
def new_key(project_id: int, body: KeyCreate, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    require_project(session, user, project_id, write=True)
    record, plaintext = create_key(session, project_id, body.name, user.id)
    return {"id": record.id, "name": record.name, "prefix": record.key_prefix,
            "key": plaintext, "note": "Copy this key now; it will not be shown again."}


@router.delete("/{key_id}")
def revoke_key(project_id: int, key_id: int, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    require_project(session, user, project_id, write=True)
    key = session.get(ProjectApiKey, key_id)
    if not key or key.project_id != project_id:
        raise HTTPException(status_code=404, detail="Key not found")
    key.revoked = True
    session.add(key)
    session.commit()
    return {"ok": True}
