"""
DecisionLog — Creates inspectable, structured decision records for OpenSmith traces and UI logging.
"""

import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class DecisionRecord(BaseModel):
    """
    Structured, human-readable record explaining runtime decision choices.
    """
    request_id: str
    timestamp: float = Field(default_factory=time.time)
    query_intent: str
    complexity_score: float
    latency_budget_ms: int
    cost_budget: float
    chosen_backend: str
    top_k: int
    reranker_enabled: bool
    selected_model: str
    reasoning_effort: str
    verification_level: str
    early_exit_triggered: bool = False
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class DecisionLogger:
    """
    Utility logger for building and logging DecisionRecords.
    """

    @staticmethod
    def create_record(
        request_id: str,
        intent: str,
        complexity: float,
        latency_budget: int,
        cost_budget: float,
        plan: Dict[str, Any],
        early_exit: bool = False
    ) -> DecisionRecord:
        return DecisionRecord(
            request_id=request_id,
            query_intent=intent,
            complexity_score=complexity,
            latency_budget_ms=latency_budget,
            cost_budget=cost_budget,
            chosen_backend=plan.get("retrieval_strategy", "hybrid"),
            top_k=plan.get("retrieval_top_k", 10),
            reranker_enabled=plan.get("reranking", True),
            selected_model=plan.get("model", "groq/openai/gpt-oss-20b"),
            reasoning_effort=plan.get("reasoning_effort", "STANDARD"),
            verification_level=plan.get("verification_level", "STANDARD"),
            early_exit_triggered=early_exit,
            rationale=plan.get("reasoning", "")
        )
