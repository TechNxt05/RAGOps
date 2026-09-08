from sqlmodel import SQLModel, Field
from typing import Optional, List
from datetime import datetime
from sqlalchemy import Column, JSON

class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    kb_version: int = Field(default=1)
    kb_version_updated_at: Optional[datetime] = Field(default=None)

class RAGConfig(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")

    # Model orchestration (ordered fallback lists)
    primary_models: List[dict] = Field(
        default=[{"provider": "groq", "name": "openai/gpt-oss-20b"}],
        sa_column=Column(JSON)
    )
    fallback_models: List[dict] = Field(
        default=[
            {"provider": "groq", "name": "llama3-8b-8192"},
            {"provider": "openrouter", "name": "meta-llama/llama-3-8b-instruct:free"},
            {"provider": "google", "name": "gemini-2.5-flash"}
        ],
        sa_column=Column(JSON)
    )
    embedding_model: str = Field(default="google-embedding-001")
    
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
    
    # Legacy field, kept for backward compat or general limits
    max_tokens: int = Field(default=2000) 

    stop_sequences: List[str] = Field(default=[], sa_column=Column(JSON))
    sensitive_topic_guard: List[str] = Field(default=[], sa_column=Column(JSON))

    # Hybrid Search Settings
    use_hybrid_search: bool = Field(default=True)
    semantic_weight: float = Field(default=0.6)  # 0.6 semantic, 0.4 BM25
    use_multi_query: bool = Field(default=True)
    enable_self_correction: bool = Field(default=True)
    enable_adaptive_router: bool = Field(default=True)
    
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

class Chunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: Optional[int] = Field(default=None, foreign_key="document.id")
    content: str
    index_id: Optional[int] = None # Mapping to FAISS ID if needed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    content_hash: Optional[str] = Field(default=None, max_length=64)
    chunk_version: int = Field(default=1)
    doc_id_version: Optional[str] = Field(default=None, max_length=256)
    chunk_index: Optional[int] = Field(default=None)


class ExternalDatabase(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    db_type: str = Field(index=True)  # postgresql, mysql, mongodb
    host: str
    port: int
    db_name: str
    username: str
    encrypted_credentials: str  # Fernet encrypted password / payload
    ssl_mode: Optional[str] = Field(default="prefer")
    schema_cache: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

