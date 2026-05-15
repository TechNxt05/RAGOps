from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class QueryLog(SQLModel, table=True):
    """Per-query analytics for RAG chat (one row per assistant turn)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    session_id: Optional[int] = Field(default=None, foreign_key="chatsession.id")

    query_text: str
    response_text: str
    model_used: str

    latency_ms: Optional[float] = None
    chunks_retrieved: int = Field(default=0)
    tokens_used: int = Field(default=0)

    citations_shown: int = Field(default=0)
    citations_clicked: int = Field(default=0)

    # hallucination_score = risk in [0,1]; lower is better (less hallucination)
    hallucination_score: Optional[float] = None
    faithfulness_score: Optional[float] = None

    context_chunks_json: Optional[str] = Field(default=None, sa_column_kwargs={"nullable": True})

    created_at: datetime = Field(default_factory=datetime.utcnow)
