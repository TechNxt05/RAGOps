"""
Test harness.

Runs against TEST_DATABASE_URL when set (CI uses Postgres + pgvector), otherwise
a throwaway SQLite file. Models and LLMs are replaced with deterministic fakes, so
no API keys, network or model downloads are needed.
"""

import hashlib
import math
import os
import re
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="ragops-test-")
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{_tmp}/test.db"
os.environ.update({
    "SECRET_KEY": "test-secret-key-not-for-production",
    "PRELOAD_MODELS": "false",
    "RUN_WORKER": "false",
    "FIRST_USER_IS_ADMIN": "true",
    "DB_CONNECTOR_ALLOW_PRIVATE_HOSTS": "true",
    "RATE_LIMIT_CHAT_PER_MIN": "1000",
    "RATE_LIMIT_UPLOAD_PER_MIN": "1000",
    "GROQ_API_KEY": "test-key",
})
os.environ.pop("ADMIN_EMAIL", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from langchain_core.language_models.chat_models import SimpleChatModel  # noqa: E402

DIM = 256
_SYNONYMS = {"vacation": "leave", "pto": "leave", "holiday": "leave", "overseas": "international", "trip": "travel", "trips": "travel"}


def fake_embed(texts):
    """Bag-of-words hashing embedder with a few synonyms, L2-normalised."""
    out = []
    for text in texts:
        vec = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            tok = _SYNONYMS.get(tok, tok)
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


class FakeChat(SimpleChatModel):
    """Answers from the CONTEXT block with a citation; writes SQL for SQL prompts."""

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _call(self, messages, stop=None, run_manager=None, **kwargs) -> str:
        text = "\n".join(str(m.content) for m in messages)
        if "SQL:" in text and "SELECT" in text.upper():
            if "per country" in text.lower():
                return ("SELECT c.country, SUM(o.amount) AS total FROM customers c JOIN orders o "
                        "ON o.customer_id = c.id GROUP BY c.country")
            return "SELECT COUNT(*) FROM customers"
        if "Results (JSON)" in text:
            return "Summary of the query results."
        if "CONTEXT:" in text:
            block = text.split("CONTEXT:", 1)[1].strip()
            m = re.search(r"\[1\] \(source: ([^)]+)\)\n(.+)", block)
            if m:
                return f"{m.group(2).split('.')[0]}. [{m.group(1)}]"
        if "Generate" in text and "search queries" in text:
            return "alternative phrasing one\nalternative phrasing two"
        if "Standalone query" in text or "standalone search query" in text:
            return "standalone question about leave"
        return "Hello! Ask me about the project's documents."


@pytest.fixture(scope="session")
def app_client():
    from app.services import llm_factory, model_registry

    model_registry.set_embedding_override(fake_embed)
    model_registry.get_reranker = lambda: None  # cosine relevance only (no 1 GB download in CI)
    llm_factory._build_chat_model = lambda *a, **k: FakeChat()
    llm_factory._model_cache.clear()

    from sqlmodel import SQLModel
    from sqlalchemy import text
    from app.db import IS_POSTGRES, engine
    if IS_POSTGRES:
        with engine.begin() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    else:
        SQLModel.metadata.drop_all(engine)

    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        yield client


def _login(client, email, password="password123"):
    r = client.post("/auth/token", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
def admin(app_client):
    r = app_client.post("/auth/register", json={"email": "admin@test.io", "password": "password123"})
    assert r.status_code == 200 and r.json()["role"] == "admin", r.text
    return _login(app_client, "admin@test.io")


@pytest.fixture(scope="session")
def employee(app_client, admin):
    r = app_client.post("/auth/register", json={"email": "emp@test.io", "password": "password123", "role": "admin"})
    assert r.status_code == 200
    return _login(app_client, "emp@test.io")


@pytest.fixture(scope="session")
def project(app_client, admin):
    from app.jobs import queue
    r = app_client.post("/rag/projects/", json={"name": "Handbook", "visibility": "all_users"}, headers=admin)
    pid = r.json()["id"]
    # The 0.40 default is calibrated for MiniLM + the BGE reranker; the bag-of-words
    # fake embedder yields lower cosines, so the test project uses its own threshold.
    cfg = app_client.patch(f"/rag/projects/{pid}/config", json={"confidence_threshold": 0.2}, headers=admin)
    assert cfg.status_code == 200 and cfg.json()["confidence_threshold"] == 0.2
    files = [
        ("files", ("leave.txt", b"Employees receive 24 days of paid annual leave per year. Up to 5 unused days carry over.", "text/plain")),
        ("files", ("travel.md", b"International travel must be approved by a director. Economy class is required on short flights.", "text/markdown")),
    ]
    r = app_client.post("/rag/ingest/upload", data={"project_id": pid}, files=files, headers=admin)
    assert r.status_code == 200, r.text
    queue.drain()
    return pid
