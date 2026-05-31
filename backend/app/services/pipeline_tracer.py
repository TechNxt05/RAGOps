"""
Pipeline failure tracer — tracks which stage caused a retrieval/generation failure.
Attached to every query, visible in admin retrieval inspector.
"""

import time
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class PipelineStage(str, Enum):
    QUERY_UNDERSTANDING = "query_understanding"
    RETRIEVAL = "retrieval"
    CONFIDENCE_GATE = "confidence_gate"
    LLM_GENERATION = "llm_generation"

class StageTrace(BaseModel):
    stage: PipelineStage
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "pending"  # pending, running, success, failed, skipped
    metadata: Dict[str, Any] = {}
    error_message: Optional[str] = None

class PipelineTracer:
    """
    Traces millisecond latency, inputs, outputs, errors,
    and metadata for each individual stage of the RAG pipeline.
    """
    def __init__(self, query: str):
        self.query = query
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.duration_ms: Optional[float] = None
        self.status = "success"
        self.traces: Dict[PipelineStage, StageTrace] = {}

    def start_stage(self, stage: PipelineStage, metadata: Optional[Dict[str, Any]] = None):
        self.traces[stage] = StageTrace(
            stage=stage,
            start_time=time.time(),
            status="running",
            metadata=metadata or {}
        )

    def end_stage(self, stage: PipelineStage, status: str = "success", metadata: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        if stage in self.traces:
            trace = self.traces[stage]
            trace.end_time = time.time()
            trace.duration_ms = (trace.end_time - trace.start_time) * 1000.0
            trace.status = status
            if metadata:
                trace.metadata.update(metadata)
            if error:
                trace.error_message = error
                self.status = "failed"
        else:
            # Fallback if start wasn't explicitly called
            now = time.time()
            self.traces[stage] = StageTrace(
                stage=stage,
                start_time=now,
                end_time=now,
                duration_ms=0.0,
                status=status,
                metadata=metadata or {},
                error_message=error
            )
            if error:
                self.status = "failed"

    def fail_pipeline(self, error: str):
        self.status = "failed"
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0

    def complete_pipeline(self):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "status": self.status,
            "total_duration_ms": self.duration_ms or ((time.time() - self.start_time) * 1000.0),
            "stages": {s.value: t.dict() for s, t in self.traces.items()}
        }
