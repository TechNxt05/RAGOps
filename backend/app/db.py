import logging
from datetime import datetime
from typing import Callable, List, Tuple

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlmodel import SQLModel, create_engine, Session

from app.core.config import settings

# Import models so SQLModel metadata registers all tables before create_all
from app.models.user import User  # noqa: F401
from app.models.workspace import Workspace, WorkspaceInvite  # noqa: F401
from app.models.rag import (  # noqa: F401
    Project, ProjectMember, RAGConfig, Document, Chunk, ChunkEmbedding,
    IngestionJob, SemanticCacheEntry, ProjectApiKey, ExternalDatabase,
)
from app.models.chat import ChatSession, Message  # noqa: F401
from app.models.usage import TokenUsage  # noqa: F401
from app.models.query_log import QueryLog  # noqa: F401

logger = logging.getLogger(__name__)

DATABASE_URL = settings.database_url
if not DATABASE_URL:
    # Local development only. Deployments must set DATABASE_URL (Neon / Postgres).
    DATABASE_URL = "sqlite:///./ragops.db"
    logger.warning("DATABASE_URL not set; using local SQLite at ./ragops.db (development only).")
elif DATABASE_URL.startswith("postgres://"):
    # Some providers still hand out the deprecated scheme SQLAlchemy rejects.
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]

IS_SQLITE = DATABASE_URL.startswith("sqlite")
IS_POSTGRES = DATABASE_URL.startswith("postgresql")

if IS_SQLITE:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=settings.db_echo)
else:
    engine = create_engine(
        DATABASE_URL,
        echo=settings.db_echo,
        pool_pre_ping=True,      # Neon closes idle connections; re-check before use
        pool_recycle=300,
        pool_size=5,
        max_overflow=10,
    )


def get_session():
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Additive schema migration
# ---------------------------------------------------------------------------

def _literal_default(column) -> str | None:
    default = column.default
    if default is None or not getattr(default, "is_scalar", False):
        return None
    value = default.arg
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return None


def _add_missing_columns(bind: Engine) -> List[str]:
    """Add columns that exist on the models but not yet in the database."""
    added = []
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())
    for table in SQLModel.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        present = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present or column.primary_key:
                continue
            col_type = column.type.compile(dialect=bind.dialect)
            ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}'
            default_sql = _literal_default(column)
            if default_sql is not None:
                ddl += f" DEFAULT {default_sql}"
            with bind.begin() as conn:
                conn.execute(text(ddl))
            added.append(f"{table.name}.{column.name}")
    return added


# ---------------------------------------------------------------------------
# One-time data migrations (tracked in schema_migrations)
# ---------------------------------------------------------------------------

def _m001_retired_models(conn) -> None:
    """Replace model IDs that providers have decommissioned."""
    import json
    from app.models.rag import DEFAULT_FALLBACK_MODELS, DEFAULT_PRIMARY_MODELS
    retired = {"llama3-70b-8192", "llama3-8b-8192", "meta-llama/llama-3-8b-instruct:free", "gemini-ultra"}
    rows = conn.execute(text("SELECT id, primary_models, fallback_models FROM ragconfig")).fetchall()
    for row_id, primary, fallback in rows:
        primary = json.loads(primary) if isinstance(primary, str) else (primary or [])
        fallback = json.loads(fallback) if isinstance(fallback, str) else (fallback or [])
        new_primary = [m for m in primary if m.get("name") not in retired] or DEFAULT_PRIMARY_MODELS
        new_fallback = [m for m in fallback if m.get("name") not in retired] or DEFAULT_FALLBACK_MODELS
        if new_primary != primary or new_fallback != fallback:
            conn.execute(
                text("UPDATE ragconfig SET primary_models = :p, fallback_models = :f WHERE id = :id"),
                {"p": json.dumps(new_primary), "f": json.dumps(new_fallback), "id": row_id},
            )


def _m002_embedding_model_name(conn) -> None:
    """'google-embedding-001' was the old default label but MiniLM was what actually
    indexed every document. Make the stored config match the vectors."""
    conn.execute(text(
        "UPDATE ragconfig SET embedding_model = 'all-MiniLM-L6-v2' "
        "WHERE embedding_model IS NULL OR embedding_model IN ('google-embedding-001', 'huggingface')"
    ))


def _m003_queue_embedding_backfill(conn) -> None:
    """Pre-pgvector deployments kept vectors in FAISS files only. Queue a
    re-index of every project that has chunks but no stored embeddings."""
    rows = conn.execute(text(
        "SELECT DISTINCT d.project_id FROM chunk c JOIN document d ON d.id = c.document_id "
        "WHERE d.project_id IS NOT NULL AND d.is_active = :t AND NOT EXISTS "
        "(SELECT 1 FROM chunkembedding ce WHERE ce.chunk_id = c.id)"
    ), {"t": True}).fetchall()
    now = datetime.utcnow()
    for (project_id,) in rows:
        conn.execute(text(
            "INSERT INTO ingestionjob (kind, status, project_id, attempts, created_at, updated_at) "
            "VALUES ('reindex_project', 'queued', :p, 0, :now, :now)"
        ), {"p": project_id, "now": now})


def _m004_default_workspace(conn) -> None:
    """Before workspaces existed every user and project shared one global space.
    Move all of it into a single workspace owned by the first admin; it keeps
    using the server's own LLM keys, exactly as before."""
    has_users = conn.execute(text("SELECT 1 FROM \"user\" LIMIT 1")).fetchone()
    has_projects = conn.execute(text("SELECT 1 FROM project LIMIT 1")).fetchone()
    if not has_users and not has_projects:
        return
    owner = conn.execute(text(
        "SELECT id FROM \"user\" WHERE LOWER(CAST(role AS VARCHAR)) = 'admin' ORDER BY id LIMIT 1"
    )).fetchone()
    conn.execute(text(
        "INSERT INTO workspace (name, owner_id, is_demo, uses_platform_keys, created_at) "
        "VALUES ('Main workspace', :owner, :f, :t, :now)"
    ), {"owner": owner[0] if owner else None, "f": False, "t": True, "now": datetime.utcnow()})
    ws_id = conn.execute(text("SELECT MAX(id) FROM workspace")).scalar()
    conn.execute(text("UPDATE \"user\" SET workspace_id = :w WHERE workspace_id IS NULL"), {"w": ws_id})
    conn.execute(text("UPDATE project SET workspace_id = :w WHERE workspace_id IS NULL"), {"w": ws_id})


MIGRATIONS: List[Tuple[str, Callable]] = [
    ("001_retired_models", _m001_retired_models),
    ("002_embedding_model_name", _m002_embedding_model_name),
    ("003_queue_embedding_backfill", _m003_queue_embedding_backfill),
    ("004_default_workspace", _m004_default_workspace),
]


def _run_data_migrations(bind: Engine) -> List[str]:
    applied = []
    with bind.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS schema_migrations (name VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP)"
        ))
        done = {r[0] for r in conn.execute(text("SELECT name FROM schema_migrations")).fetchall()}
    for name, fn in MIGRATIONS:
        if name in done:
            continue
        with bind.begin() as conn:
            fn(conn)
            conn.execute(
                text("INSERT INTO schema_migrations (name, applied_at) VALUES (:n, :t)"),
                {"n": name, "t": datetime.utcnow()},
            )
        applied.append(name)
    return applied


def init_db() -> None:
    """Create/upgrade the schema. Fails loudly: a misconfigured database must
    never silently fall back to a throwaway local file in production."""
    if IS_POSTGRES:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    SQLModel.metadata.create_all(engine)
    added = _add_missing_columns(engine)
    if added:
        logger.info("Schema upgraded, added columns: %s", ", ".join(added))
    applied = _run_data_migrations(engine)
    if applied:
        logger.info("Applied data migrations: %s", ", ".join(applied))
