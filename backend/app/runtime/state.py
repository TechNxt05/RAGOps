"""
RuntimeState — Typed Pydantic v2 state object tracking request lifecycle signals,
complexity scores, budgets, selected strategies, and execution history.
"""

import time
import uuid
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    FACTOID = "factoid"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    DEFINITIONAL = "definitional"
    EXACT_LOOKUP = "exact_lookup"
    STRUCTURED = "structured"
    MULTI_HOP = "multi_hop"
    CHIT_CHAT = "chit_chat"
    UNKNOWN = "unknown"


class CacheStatus(str, Enum):
    NONE = "none"
    PROMPT_HIT = "prompt_hit"
    SEMANTIC_HIT = "semantic_hit"
    SESSION_HIT = "session_hit"
    BYPASS = "bypass"


class RuntimeState(BaseModel):
    """
    Central typed runtime state representing a single request execution pass.
    """
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    session_id: str = Field(default="default_session")
    project_id: int = Field(default=1)

    query: str
    normalized_query: str = ""

    intent: QueryIntent = QueryIntent.UNKNOWN
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    ambiguity: float = Field(default=0.0, ge=0.0, le=1.0)
    entity_density: float = Field(default=0.0, ge=0.0, le=1.0)
    temporal_requirement: bool = False
    mathematical_requirement: bool = False
    multi_hop_requirement: bool = False

    retrieval_confidence: Optional[float] = None
    generation_confidence: Optional[float] = None
    verification_confidence: Optional[float] = None

    latency_budget_ms: int = Field(default=3000, gt=0)
    cost_budget: float = Field(default=0.02, ge=0.0)
    token_budget: int = Field(default=4096, gt=0)
    quality_target: float = Field(default=0.85, ge=0.0, le=1.0)

    selected_retrieval_backend: str = "hybrid"
    selected_model: str = "groq/llama3-70b-8192"
    selected_context_strategy: str = "standard"

    retry_count: int = 0
    repair_count: int = 0
    cache_status: CacheStatus = CacheStatus.NONE

    execution_plan: Optional[Dict[str, Any]] = None
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    final_status: str = "initialized"
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None

    def elapsed_ms(self) -> float:
        """Returns elapsed milliseconds since state initialization."""
        end = self.end_time if self.end_time else time.time()
        return round((end - self.start_time) * 1000, 2)

    def add_trace_event(self, stage: str, details: Dict[str, Any]) -> None:
        """Appends a structured event to the runtime execution trace."""
        self.execution_trace.append({
            "stage": stage,
            "timestamp": time.time(),
            "elapsed_ms": self.elapsed_ms(),
            "details": details
        })
