from typing import Any, List, Tuple


def _doc_key(doc: Any) -> str:
    meta = getattr(doc, "metadata", None) or {}
    if meta.get("chunk_id") is not None:
        return f"chunk:{meta['chunk_id']}"
    content = doc.page_content if hasattr(doc, "page_content") else str(doc)
    # Keep first 500 chars as key to avoid oversized keys
    return "text:" + content[:500]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Any, float]]],
    k: int = 60,
    weights: List[float] = None
) -> List[Tuple[Any, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ weight_i / (k + rank_i)

    Args:
        ranked_lists: List of ranked result lists, each item is (document, score)
        k: Constant to prevent high scores for top-ranked items (default 60)
        weights: Weight per list (default: equal weights)

    Returns:
        Merged list of (document, rrf_score) sorted by rrf_score descending
    """
    if not ranked_lists:
        return []

    if weights is None or len(weights) != len(ranked_lists):
        weights = [1.0] * len(ranked_lists)

    # Normalize weights
    total_weight = sum(weights) or 1.0
    weights = [w / total_weight for w in weights]

    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Any] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (doc, _score) in enumerate(ranked_list):
            key = _doc_key(doc)
            doc_map.setdefault(key, doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + weight / (k + rank + 1)

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[key], score) for key, score in sorted_docs]


def hybrid_search_merge(
    semantic_results: List[Tuple[Any, float]],
    bm25_results: List[Tuple[Any, float]],
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
    k: int = 60,
) -> List[Tuple[Any, float]]:
    """
    Merge semantic (vector) and keyword (BM25) result lists with weighted RRF.
    Only ranks are used, so the different score scales don't matter.
    bm25_results may hold LCDocuments or raw chunk texts.
    """
    from langchain_core.documents import Document as LCDocument

    formatted_bm25 = []
    for item, score in bm25_results:
        if isinstance(item, str):
            item = LCDocument(page_content=item, metadata={"source": "Lexical Search (BM25)"})
        formatted_bm25.append((item, score))

    return reciprocal_rank_fusion(
        ranked_lists=[semantic_results, formatted_bm25],
        k=k,
        weights=[semantic_weight, bm25_weight]
    )
