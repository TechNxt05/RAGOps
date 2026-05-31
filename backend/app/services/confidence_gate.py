"""
Source Confidence Scoring — pre-generation quality gate.
Computes retrieval confidence and refuses to generate when evidence is weak.
This is what 'zero hallucination' actually means in production:
not that the model never tries, but that we refuse when retrieval confidence is low.
"""

from dataclasses import dataclass
from typing import List, Optional
import time

@dataclass 
class ChunkConfidence:
    chunk_id: str
    chunk_text: str
    retrieval_score: float      # From reranker (0-1)
    freshness_score: float      # Based on document age
    authority_score: float      # Based on document type/source trust
    agreement_score: float      # Agreement with other retrieved chunks
    
    @property
    def confidence(self) -> float:
        return (
            self.retrieval_score * 0.50 +
            self.freshness_score * 0.20 +
            self.authority_score * 0.20 +
            self.agreement_score * 0.10
        )

@dataclass
class ConfidenceGateResult:
    passed: bool                        # True = proceed to generation
    max_confidence: float               # Best chunk confidence
    avg_confidence: float               # Average across all chunks
    low_confidence_chunks: int          # Chunks below threshold
    refusal_reason: Optional[str]       # Why generation was refused
    chunk_confidences: List[ChunkConfidence]

CONFIDENCE_THRESHOLD = 0.65             # From paper recommendation
AUTHORITY_SCORES = {
    "pdf": 0.8,
    "docx": 0.75,
    "txt": 0.6,
    "md": 0.65,
    "html": 0.5,
    "unknown": 0.5,
}

class SourceConfidenceGate:
    """
    Pre-generation confidence gate.
    Call AFTER retrieval/reranking, BEFORE LLM generation.
    """
    
    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold
    
    def evaluate(
        self,
        chunks: List[dict],             # Retrieved chunks with metadata
        reranker_scores: List[float],   # BGE reranker scores (aligned with chunks)
        document_metadata: List[dict],  # File type, upload date per chunk
    ) -> ConfidenceGateResult:
        
        chunk_confidences = []
        
        for i, (chunk, rerank_score) in enumerate(zip(chunks, reranker_scores)):
            meta = document_metadata[i] if i < len(document_metadata) else {}
            
            # Retrieval score: normalize reranker score to 0-1
            # Reranker score is usually a float, sometimes positive/negative (e.g. -2.5 to 5.0)
            # Let's map it safely
            retrieval_score = max(0.0, min(1.0, (rerank_score + 10) / 20))
            
            # Freshness: documents uploaded recently score higher
            upload_date = meta.get("upload_date")
            freshness_score = self._compute_freshness(upload_date)
            
            # Authority: based on document type
            file_type = meta.get("file_type", "unknown").lower().lstrip(".")
            authority_score = AUTHORITY_SCORES.get(file_type, 0.5)
            
            # Agreement: how much does this chunk agree with others?
            agreement_score = self._compute_agreement(
                chunk.get("text", chunk.get("content", "")), 
                [c.get("text", c.get("content", "")) for j, c in enumerate(chunks) if j != i]
            )
            
            chunk_confidences.append(ChunkConfidence(
                chunk_id=str(chunk.get("id", i)),
                chunk_text=chunk.get("text", chunk.get("content", ""))[:200],
                retrieval_score=retrieval_score,
                freshness_score=freshness_score,
                authority_score=authority_score,
                agreement_score=agreement_score,
            ))
        
        if not chunk_confidences:
            return ConfidenceGateResult(
                passed=False,
                max_confidence=0.0,
                avg_confidence=0.0,
                low_confidence_chunks=0,
                refusal_reason="No chunks retrieved",
                chunk_confidences=[],
            )
        
        confidences = [c.confidence for c in chunk_confidences]
        max_conf = max(confidences)
        avg_conf = sum(confidences) / len(confidences)
        low_conf_count = sum(1 for c in confidences if c < self.threshold)
        
        # Gate logic: if BEST chunk confidence is below threshold → refuse
        passed = max_conf >= self.threshold
        refusal_reason = None
        if not passed:
            refusal_reason = (
                f"Retrieved evidence has insufficient confidence "
                f"(best: {max_conf:.2f}, threshold: {self.threshold:.2f}). "
                f"The knowledge base may not contain reliable information for this query."
            )
        
        return ConfidenceGateResult(
            passed=passed,
            max_confidence=max_conf,
            avg_confidence=avg_conf,
            low_confidence_chunks=low_conf_count,
            refusal_reason=refusal_reason,
            chunk_confidences=chunk_confidences,
        )
    
    def _compute_freshness(self, upload_date) -> float:
        if upload_date is None:
            return 0.5
        try:
            now = time.time()
            if hasattr(upload_date, 'timestamp'):
                age_days = (now - upload_date.timestamp()) / 86400
            elif isinstance(upload_date, (int, float)):
                age_days = (now - upload_date) / 86400
            elif isinstance(upload_date, str):
                from datetime import datetime
                # Handle isoformat or simple date
                try:
                    dt = datetime.fromisoformat(upload_date.replace("Z", "+00:00"))
                    age_days = (datetime.utcnow() - dt.replace(tzinfo=None)).days
                except:
                    age_days = 30
            else:
                age_days = 30  # Default assumption
            # Freshness decays over 365 days
            return max(0.1, 1.0 - (age_days / 365))
        except:
            return 0.5
    
    def _compute_agreement(self, chunk_text: str, other_texts: List[str]) -> float:
        """Measure lexical agreement between this chunk and others"""
        if not other_texts:
            return 0.5
        
        chunk_words = set(chunk_text.lower().split())
        agreements = []
        for other in other_texts[:5]:  # Compare with up to 5 others
            other_words = set(other.lower().split())
            if not other_words:
                continue
            # Jaccard similarity
            intersection = chunk_words & other_words
            union = chunk_words | other_words
            if union:
                agreements.append(len(intersection) / len(union))
        
        return sum(agreements) / len(agreements) if agreements else 0.5


def get_confidence_gate(threshold: float = CONFIDENCE_THRESHOLD) -> SourceConfidenceGate:
    return SourceConfidenceGate(threshold=threshold)
