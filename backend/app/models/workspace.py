from datetime import datetime
from typing import List, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class Workspace(SQLModel, table=True):
    """A tenant: one company/team. Every user and project belongs to exactly one
    workspace, and nothing is ever visible across workspaces."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=200)
    owner_id: Optional[int] = Field(default=None, index=True)
    # The public demo workspace (one-click demo logins, restricted, reset daily).
    is_demo: bool = Field(default=False)
    # True: LLM calls use the server's own keys (GROQ_API_KEY, ...). Only the
    # platform owner's workspace and the demo get this; everyone else brings
    # their own keys.
    uses_platform_keys: bool = Field(default=False)
    # Fernet-encrypted JSON {"groq": "...", "google": "...", "openrouter": "..."}.
    llm_keys_encrypted: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WorkspaceInvite(SQLModel, table=True):
    """An admin's invitation for one email. Joining needs the secret link the
    admin shares (emails aren't verified, so the email alone proves nothing).
    Only a SHA-256 hash of the token is stored."""
    id: Optional[int] = Field(default=None, primary_key=True)
    workspace_id: int = Field(foreign_key="workspace.id", index=True)
    email: str = Field(index=True, max_length=320)
    token_hash: str = Field(default="", index=True, max_length=64)
    role: str = Field(default="client", max_length=16)
    project_ids: List[int] = Field(default_factory=list, sa_column=Column(JSON))
    created_by: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
