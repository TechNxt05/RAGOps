"""
Lightweight RAG quality signals using TF-IDF cosine similarity and token overlap.
Hallucination score is stored as *risk* (0 = grounded, 1 = likely ungrounded).
"""

from __future__ import annotations

from typing import List, Dict

from app.utils.tfidf import cosine, tfidf_vectors


def _max_grounding_similarity(response: str, context_chunks: List[str]) -> float:
    """How well the response aligns with at least one chunk (higher = more grounded)."""
    if not response.strip() or not context_chunks:
        return 0.0

    corpus = context_chunks + [response]
    try:
        vectors = tfidf_vectors(corpus, stop_words="english", max_features=500)
        similarities = [cosine(vectors[-1], v) for v in vectors[:-1]]
        return max(similarities) if similarities else 0.5
    except Exception:
        return 0.0


def score_faithfulness(response: str, context_chunks: List[str]) -> float:
    """Fraction of response token types that appear in retrieved context."""
    if not response.strip() or not context_chunks:
        return 0.0

    context_text = " ".join(context_chunks).lower()
    response_words = {w for w in response.lower().split() if w.isalnum() or len(w) > 1}
    context_words = set(context_text.split())

    if not response_words:
        return 0.0

    overlap = len(response_words & context_words)
    return min(1.0, overlap / len(response_words))


def evaluate_rag_response(
    query: str,
    response: str,
    context_chunks: List[str],
) -> Dict[str, float | str]:
    grounding = _max_grounding_similarity(response, context_chunks)
    faithfulness = score_faithfulness(response, context_chunks)
    hallucination_risk = max(0.0, min(1.0, 1.0 - grounding))

    # Calculate context relevance (query vs chunks)
    context_relevance = _max_grounding_similarity(query, context_chunks) if context_chunks else 0.0

    # Calculate answer relevance (query vs response)
    answer_relevance = 0.0
    if query.strip() and response.strip():
        q_words = set(w.lower() for w in query.split() if len(w) > 2)
        r_words = set(w.lower() for w in response.split() if len(w) > 2)
        if q_words and r_words:
            overlap = len(q_words & r_words) / len(q_words)
            answer_relevance = min(1.0, max(0.1, overlap * 1.5))

    overall = context_relevance * 0.25 + faithfulness * 0.35 + answer_relevance * 0.25 + grounding * 0.15

    label = (
        "EXCELLENT"
        if overall > 0.8
        else "GOOD"
        if overall > 0.6
        else "FAIR"
        if overall > 0.4
        else "POOR"
    )

    return {
        "context_relevance": context_relevance,
        "faithfulness": faithfulness,
        "faithfulness_score": faithfulness,
        "answer_relevance": answer_relevance,
        "groundedness": grounding,
        "hallucination_score": hallucination_risk,
        "overall_quality_score": overall,
        "quality_label": label,
    }
