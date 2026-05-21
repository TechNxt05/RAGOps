from typing import Any, List, Tuple

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

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        weights = [1.0] * len(ranked_lists)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Build RRF scores
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Any] = {}

    for list_idx, (ranked_list, weight) in enumerate(zip(ranked_lists, weights)):
        for rank, (doc, _score) in enumerate(ranked_list):
            # Use doc page_content or str representation as key for deduplication
            if hasattr(doc, 'page_content'):
                doc_key = doc.page_content
            else:
                doc_key = str(doc)
            
            # Keep first 500 chars as key to avoid oversized keys
            doc_key = doc_key[:500]
            doc_map[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (
                weight / (k + rank + 1)
            )

    # Sort by RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [(doc_map[key], score) for key, score in sorted_docs]


def hybrid_search_merge(
    semantic_results: List[Tuple[Any, float]],  # (chunk, cosine_score)
    bm25_results: List[Tuple[str, float]],      # (chunk_text, bm25_score)
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
    k: int = 60,
) -> List[Tuple[Any, float]]:
    """
    Merge semantic (FAISS) and keyword (BM25) results using RRF.

    Default weights: 60% semantic, 40% BM25.
    """
    # FAISS score is L2 distance or similar, but for ranking, rank is order in list.
    # BM25 score is raw BM25 score.
    # The reciprocal_rank_fusion only uses the order/rank of items, which is perfect!
    
    # We must convert bm25_results (which are (str, float)) into the same format/objects as FAISS (LCDocument),
    # or keep them as whatever they are and map them during RRF.
    # Let's map BM25 text results to the exact same LCDocument objects if there is a content match,
    # so we don't have duplicate representations.
    
    # Create a mapping from content to LCDocument from semantic_results
    doc_by_content = {}
    for doc, _ in semantic_results:
        doc_by_content[doc.page_content] = doc

    # Format BM25 results: if we have the LCDocument from semantic search, use it;
    # otherwise, create a mock LCDocument for it so it contains the page_content and source metadata.
    from langchain_core.documents import Document as LCDocument
    
    formatted_bm25 = []
    for text, score in bm25_results:
        if text in doc_by_content:
            formatted_bm25.append((doc_by_content[text], score))
        else:
            # We build a document. We don't have full metadata here, but we can search for it if needed.
            # However, a simple LCDocument is enough.
            new_doc = LCDocument(page_content=text, metadata={"source": "Lexical Search (BM25)"})
            formatted_bm25.append((new_doc, score))

    return reciprocal_rank_fusion(
        ranked_lists=[semantic_results, formatted_bm25],
        k=k,
        weights=[semantic_weight, bm25_weight]
    )
