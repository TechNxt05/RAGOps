"""
Lightweight RAG quality signals using TF-IDF cosine similarity and token overlap.
Hallucination score is stored as *risk* (0 = grounded, 1 = likely ungrounded).
"""

from __future__ import annotations

from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _max_grounding_similarity(response: str, context_chunks: List[str]) -> float:
    """How well the response aligns with at least one chunk (higher = more grounded)."""
    if not context_chunks or not response.strip():
        return 0.5

    corpus = context_chunks + [response]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)

    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        response_vec = tfidf_matrix[-1]
        context_vecs = tfidf_matrix[:-1]
        similarities = cosine_similarity(response_vec, context_vecs)[0]
        return float(np.max(similarities)) if similarities.size else 0.5
    except Exception:
        return 0.5


def score_faithfulness(response: str, context_chunks: List[str]) -> float:
    """Fraction of response token types that appear in retrieved context."""
    if not context_chunks or not response.strip():
        return 0.5

    context_text = " ".join(context_chunks).lower()
    response_words = {w for w in response.lower().split() if w.isalnum() or len(w) > 1}
    context_words = set(context_text.split())

    if not response_words:
        return 0.5

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
    overall = grounding * 0.6 + faithfulness * 0.4

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
        "hallucination_score": hallucination_risk,
        "faithfulness_score": faithfulness,
        "overall_quality_score": overall,
        "quality_label": label,
    }
