from typing import List, Tuple, Any

from app.services import model_registry


def _text_of(c: Any) -> str:
    if hasattr(c, 'page_content'):
        return c.page_content
    if isinstance(c, tuple) and len(c) > 0 and hasattr(c[0], 'page_content'):
        return c[0].page_content
    if isinstance(c, dict) and 'content' in c:
        return c['content']
    return str(c)


class BGEReranker:
    """
    Reranks chunks with the BAAI/bge-reranker-base cross-encoder.

    For (document, score) tuples the score is REPLACED by the cross-encoder's
    relevance probability in [0, 1], so downstream stages (confidence gate,
    compression, UI) see a real relevance signal instead of the RRF/vector score
    the chunk arrived with. Falls back to the incoming order if the model is
    unavailable.
    """

    def __init__(self):
        self.last_scored = False

    @property
    def enabled(self) -> bool:
        return model_registry.get_reranker() is not None

    def initialize(self):
        model_registry.get_reranker()

    def rerank(
        self,
        query: str,
        chunks: List[Any],
        top_k: int = 5
    ) -> List[Any]:
        out, scored = self.rerank_scored(query, chunks, top_k)
        self.last_scored = scored
        return out

    def rerank_scored(self, query: str, chunks: List[Any], top_k: int = 5) -> Tuple[List[Any], bool]:
        """Thread-safe variant: returns (results, whether cross-encoder scores were applied)."""
        if not chunks:
            return [], False
        # Candidates arrive best-first from retrieval; cross-encoding is the costliest
        # CPU step, so only the top RERANK_MAX_CANDIDATES are scored.
        from app.core.config import settings
        chunks = chunks[:max(settings.rerank_max_candidates, top_k)]
        try:
            scores = model_registry.rerank_scores(query, [_text_of(c) for c in chunks])
        except Exception as e:
            print(f"Error in reranking: {e}. Falling back to default order.")
            scores = None
        if scores is None:
            return chunks[:top_k], False

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for chunk, score in ranked:
            if isinstance(chunk, tuple) and len(chunk) == 2:
                doc = chunk[0]
                if hasattr(doc, "metadata"):
                    doc.metadata["rerank_score"] = round(score, 4)
                out.append((doc, score))
            else:
                out.append(chunk)
        return out, True

    def adaptive_rerank(
        self,
        query: str,
        chunks: List[Any],
        confidence: float = 0.50,
        top_k: int = 5,
        allow_rerank_skip: bool = True
    ) -> Tuple[List[Any], dict]:
        """
        Adaptive reranking policy decision:
        - confidence >= 0.85 and allow_rerank_skip: skip cross-encoder reranking (save latency & cost)
        - confidence >= 0.50: standard rerank
        - confidence < 0.50: expand top_k + rerank
        Returns (reranked_chunks, stats_dict).
        """
        if confidence >= 0.85 and allow_rerank_skip:
            return chunks[:top_k], {
                "reranked": False,
                "reason": "HIGH initial confidence (>=0.85): skipped expensive cross-encoder reranker",
                "avoided_reranker_cost_tokens": len(chunks) * 50
            }

        effective_k = top_k if confidence >= 0.50 else max(top_k + 3, 10)
        reranked = self.rerank(query, chunks, top_k=effective_k)
        return reranked, {
            "reranked": True,
            "reason": "MEDIUM/LOW confidence: cross-encoder reranked candidate set",
            "avoided_reranker_cost_tokens": 0
        }

# Singleton instance
reranker_service = BGEReranker()
