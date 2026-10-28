from sqlmodel import SQLModel, Field
from typing import Optional, List
from datetime import datetime
from sqlalchemy import Column, JSON, LargeBinary, ForeignKey, Integer
from pgvector.sqlalchemy import Vector

# Live model IDs (verified against provider model lists, Oct 2026). The UI pulls
# the authoritative list from GET /models/catalog; these are only defaults.
DEFAULT_PRIMARY_MODELS = [{"provider": "groq", "name": "openai/gpt-oss-20b"}]
DEFAULT_FALLBACK_MODELS = [
    {"provider": "groq", "name": "qwen/qwen3.8-27b"},
    {"provider": "google", "name": "gemini-2.5-flash"},
]


class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    kb_version: int = Field(default=1)
    kb_version_updated_at: Optional[datetime] = Field(default=None)
    # "all_users": every signed-in user may chat with the project (legacy behaviour).
    # "members_only": only admins and users listed in ProjectMember.
    visibility: str = Field(default="all_users", max_length=32)
    owner_id: Optional[int] = Field(default=None, foreign_key="user.id")


class ProjectMember(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    # "viewer" can chat/search; "editor" can also upload and manage documents.
    role: str = Field(default="viewer", max_length=16)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RAGConfig(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")

    # Model orchestration (ordered fallback lists)
    primary_models: List[dict] = Field(
        default_factory=lambda: [dict(m) for m in DEFAULT_PRIMARY_MODELS],
        sa_column=Column(JSON)
    )
    fallback_models: List[dict] = Field(
        default_factory=lambda: [dict(m) for m in DEFAULT_FALLBACK_MODELS],
        sa_column=Column(JSON)
    )
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # Text Splitter Config (Existing)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    # Generation Controls
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)
    max_output_tokens: int = Field(default=1024)
    response_style: str = Field(default="Concise")

    # Retrieval Controls
    top_k: int = Field(default=4)
    similarity_threshold: float = Field(default=0.0)
    max_context_tokens: int = Field(default=2048)
    context_ordering: str = Field(default="similarity") # similarity, chronological

    # Safety Controls
    answer_only_from_docs: bool = Field(default=False)
    hallucination_guard: bool = Field(default=False)
    confidence_threshold: float = Field(default=0.4)  # calibrated, see model_registry

    # Legacy field, kept for backward compat or general limits
    max_tokens: int = Field(default=2000)

    stop_sequences: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    sensitive_topic_guard: List[str] = Field(default_factory=list, sa_column=Column(JSON))

    # Hybrid Search Settings
    use_hybrid_search: bool = Field(default=True)
    semantic_weight: float = Field(default=0.6)  # 0.6 semantic, 0.4 BM25
    use_multi_query: bool = Field(default=True)
    enable_self_correction: bool = Field(default=True)
    enable_adaptive_router: bool = Field(default=True)

    # Per-technique toggles (every stage of the pipeline is switchable per project)
    enable_query_rewrite: bool = Field(default=True)
    enable_hyde: bool = Field(default=False)
    enable_reranker: bool = Field(default=True)
    enable_cross_reference: bool = Field(default=True)
    enable_conflict_detection: bool = Field(default=True)
    enable_compression: bool = Field(default=True)
    enable_claim_verification: bool = Field(default=True)
    enable_reflective_generation: bool = Field(default=False)
    enable_semantic_cache: bool = Field(default=True)
    semantic_cache_threshold: float = Field(default=0.92)
    # Tier-based model routing (cheap model for simple questions). Off by default:
    # when off, the model the user/project picked is always the one used.
    enable_cost_routing: bool = Field(default=False)

    # Phase 3: Admin Pipeline Selection
    pipeline_variant: str = Field(default="naive")

    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")
    filename: str
    content: str = Field(sa_column_kwargs={"nullable": True})
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = Field(default=False)

    file_size_bytes: Optional[int] = None
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    chunk_size_used: Optional[int] = None
    embedding_model_used: Optional[str] = None
    version: int = Field(default=1)
    is_active: bool = Field(default=True)
    processing_status: str = Field(default="pending")
    processing_error: Optional[str] = None
    uploaded_by: Optional[int] = Field(default=None, foreign_key="user.id")

    chunking_strategy: Optional[str] = Field(default=None)
    chunking_metrics: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    document_hash: Optional[str] = Field(default=None, max_length=64)
    parsing_method: Optional[str] = Field(default=None, max_length=64)
    redaction_log: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    parsed_chunks_json: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    ingestion_scan: Optional[dict] = Field(default=None, sa_column=Column(JSON))

class Chunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: Optional[int] = Field(default=None, foreign_key="document.id", index=True)
    content: str
    index_id: Optional[int] = None # Legacy FAISS mapping, unused since pgvector
    created_at: datetime = Field(default_factory=datetime.utcnow)
    content_hash: Optional[str] = Field(default=None, max_length=64)
    chunk_version: int = Field(default=1)
    doc_id_version: Optional[str] = Field(default=None, max_length=256)
    chunk_index: Optional[int] = Field(default=None)
    chunk_metadata: Optional[dict] = Field(default=None, sa_column=Column(JSON))


class ChunkEmbedding(SQLModel, table=True):
    """
    One vector per (chunk, embedding model). Stored in Postgres via pgvector so
    vectors live next to the rows they describe: per-project filtering happens
    in SQL, writes are transactional, and nothing is lost on an ephemeral disk.
    The column is dimension-less so projects can use different embedding models;
    per-dimension partial HNSW indexes are created on demand (see vector_store).
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    chunk_id: int = Field(sa_column=Column(Integer, ForeignKey("chunk.id", ondelete="CASCADE"), index=True, nullable=False))
    document_id: int = Field(index=True)
    project_id: int = Field(index=True)
    model: str = Field(max_length=128, index=True)
    dim: int
    embedding: List[float] = Field(sa_column=Column(Vector().with_variant(JSON(), "sqlite"), nullable=False))


class IngestionJob(SQLModel, table=True):
    """Durable work queue (Postgres-backed) for parsing, indexing and re-indexing."""
    id: Optional[int] = Field(default=None, primary_key=True)
    kind: str = Field(max_length=32, index=True)  # ingest_file | reindex_document | reindex_project
    status: str = Field(default="queued", max_length=16, index=True)  # queued | running | done | failed
    project_id: Optional[int] = Field(default=None, index=True)
    document_id: Optional[int] = Field(default=None, index=True)
    payload: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    file_bytes: Optional[bytes] = Field(default=None, sa_column=Column(LargeBinary, nullable=True))
    attempts: int = Field(default=0)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SemanticCacheEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(index=True)
    kb_version: int = Field(index=True)
    model: str = Field(max_length=128)
    query: str
    keywords: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    embedding: List[float] = Field(sa_column=Column(JSON))
    response: dict = Field(sa_column=Column(JSON))
    hits: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ProjectApiKey(SQLModel, table=True):
    """API keys for programmatic / MCP access to a single project."""
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    name: str = Field(max_length=128)
    key_prefix: str = Field(max_length=16, index=True)
    key_hash: str = Field(max_length=128)
    created_by: Optional[int] = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    revoked: bool = Field(default=False)


class ExternalDatabase(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    name: Optional[str] = Field(default=None, max_length=128)
    description: Optional[str] = None
    db_type: str = Field(index=True)  # postgresql, mysql
    host: str
    port: int
    db_name: str
    username: str
    encrypted_credentials: str  # Fernet encrypted password / payload
    ssl_mode: Optional[str] = Field(default="prefer")
    schema_cache: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    # Tables the manager has exposed to employees. Empty list = nothing exposed.
    allowed_tables: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
