"""
Which LLM API keys the current request may use.

Each workspace brings its own provider keys; only the platform owner's workspace
and the public demo use the server's environment keys. The keys travel with the
request in a context variable, so every LLM call site (chat, rewriting, SQL
agent, benchmark...) picks the right ones without threading them through.

Context variables are inherited by asyncio tasks and by run_in_threadpool, so a
scope set in the request's auth dependency covers the whole request. Background
workers set it per job (see app.jobs.queue).
"""

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, Optional

from fastapi import HTTPException

ENV_NAMES = {
    "groq": "GROQ_API_KEY",
    "google": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
# Providers a workspace admin can configure in the UI (all have free tiers).
WORKSPACE_PROVIDERS = ("groq", "google", "openrouter")

NO_KEYS_MESSAGE = (
    "No LLM API key is configured for this workspace. An admin can add a free "
    "Groq, Gemini or OpenRouter key under Workspace settings."
)

# None = no workspace scope (CLI scripts, tests, the platform's own workspace):
# use the environment. A dict = exactly these keys and nothing else.
_scope: ContextVar[Optional[Dict[str, str]]] = ContextVar("llm_keys", default=None)


class MissingLLMKeys(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=400, detail=NO_KEYS_MESSAGE)


def get_key(provider: str) -> Optional[str]:
    keys = _scope.get()
    if keys is None:
        env = ENV_NAMES.get(provider)
        return os.getenv(env) if env else None
    return keys.get(provider) or None


def fingerprint() -> str:
    """Distinguishes key sets in caches without storing the keys twice."""
    import hashlib
    keys = _scope.get()
    raw = "env" if keys is None else json.dumps(keys, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def decrypt_keys(encrypted: Optional[str]) -> Dict[str, str]:
    if not encrypted:
        return {}
    from app.services.db_connector.credential_vault import credential_vault
    try:
        data = json.loads(credential_vault.decrypt(encrypted))
    except Exception:
        # SECRET_KEY changed since the keys were saved: they must be re-entered.
        return {}
    return {p: v for p, v in data.items() if p in WORKSPACE_PROVIDERS and v}


def encrypt_keys(keys: Dict[str, str]) -> Optional[str]:
    clean = {p: v.strip() for p, v in keys.items() if p in WORKSPACE_PROVIDERS and v and v.strip()}
    if not clean:
        return None
    from app.services.db_connector.credential_vault import credential_vault
    return credential_vault.encrypt(json.dumps(clean))


def keys_for_workspace(workspace) -> Optional[Dict[str, str]]:
    """The scope value for a workspace (None means 'use the environment')."""
    if workspace is None:
        return {}
    if workspace.uses_platform_keys:
        return None
    return decrypt_keys(workspace.llm_keys_encrypted)


def set_scope(keys: Optional[Dict[str, str]]):
    return _scope.set(keys)


@contextmanager
def scope(keys: Optional[Dict[str, str]]) -> Iterator[None]:
    token = _scope.set(keys)
    try:
        yield
    finally:
        _scope.reset(token)


def scope_for_project(session, project_id: Optional[int]) -> Optional[Dict[str, str]]:
    """Key scope for work done on behalf of a project (background jobs, MCP)."""
    from app.models.rag import Project
    from app.models.workspace import Workspace
    project = session.get(Project, project_id) if project_id else None
    if project is None:
        return {}
    workspace = session.get(Workspace, project.workspace_id) if project.workspace_id else None
    return keys_for_workspace(workspace)


def mask(key: str) -> str:
    return (key[:4] + "…" + key[-4:]) if len(key) > 10 else "…"
