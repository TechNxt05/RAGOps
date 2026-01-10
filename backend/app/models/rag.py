from sqlmodel import SQLModel, Field
from typing import Optional, List
from datetime import datetime
from sqlalchemy import Column, JSON

class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RAGConfig(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")
    
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

    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")
    filename: str
    content: str = Field(sa_column_kwargs={"nullable": True})
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = Field(default=False)

class Chunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: Optional[int] = Field(default=None, foreign_key="document.id")
    content: str
    index_id: Optional[int] = None # Mapping to FAISS ID if needed
    created_at: datetime = Field(default_factory=datetime.utcnow)
