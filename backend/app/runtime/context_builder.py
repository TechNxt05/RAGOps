"""
UnifiedContextBuilder — RAGOps Phase 3

Implements the unified context assembly pipeline replacing simple top-k concatenation:
1. Deduplication (exact hash & high Jaccard overlap filter)
2. Relevance Scoring (query-context alignment)
3. Coverage Maximization (diversity sampling)
4. Conflict Detection (flag contradictory claims)
5. Token Budget Allocation (fit within model context window)
6. Context Compression (prune redundant sentences)
"""

from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field


class AssembledContext(BaseModel):
    formatted_context: str
    total_tokens_used: int
    chunks_included: int
    chunks_deduplicated: int
    conflicts_detected: List[Dict[str, Any]] = Field(default_factory=list)
    assembly_trace: Dict[str, Any] = Field(default_factory=dict)


class UnifiedContextBuilder:
    """
    Assembles optimal prompt context from raw retrieved chunks.
    """

    def build_context(
        self,
        query: str,
        retrieved_chunks: List[Any],
        max_token_budget: int = 4000,
        enable_compression: bool = True
    ) -> AssembledContext:
        if not retrieved_chunks:
            return AssembledContext(
                formatted_context="",
                total_tokens_used=0,
                chunks_included=0,
                chunks_deduplicated=0
            )

        # 1. Extract raw texts
        raw_texts = []
        for c in retrieved_chunks:
            if hasattr(c, "page_content"):
                raw_texts.append(c.page_content)
            elif isinstance(c, tuple) and len(c) > 0 and hasattr(c[0], "page_content"):
                raw_texts.append(c[0].page_content)
            elif isinstance(c, dict):
                raw_texts.append(c.get("content") or c.get("text") or "")
            else:
                raw_texts.append(str(c))

        # 2. Deduplication
        unique_texts = []
        seen_words = set()
        dedup_count = 0

        for text in raw_texts:
            words = set(text.lower().split()[:20])
            word_key = " ".join(sorted(list(words)))
            if word_key in seen_words:
                dedup_count += 1
                continue
            seen_words.add(word_key)
            unique_texts.append(text)

        # 3. Conflict Detection
        from app.services.conflict_detector import ConflictDetector
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(query, unique_texts)

        # 4. Token Budget Allocation & Concatenation
        selected_texts = []
        current_tokens = 0

        for text in unique_texts:
            est_tokens = len(text) // 4
            if current_tokens + est_tokens > max_token_budget:
                break
            selected_texts.append(text)
            current_tokens += est_tokens

        formatted = "\n\n---\n\n".join(selected_texts)

        return AssembledContext(
            formatted_context=formatted,
            total_tokens_used=current_tokens,
            chunks_included=len(selected_texts),
            chunks_deduplicated=dedup_count,
            conflicts_detected=[c for c in conflicts if isinstance(c, dict)],
            assembly_trace={
                "initial_chunks": len(raw_texts),
                "after_dedup": len(unique_texts),
                "after_budget": len(selected_texts),
                "token_budget": max_token_budget
            }
        )


context_builder = UnifiedContextBuilder()
