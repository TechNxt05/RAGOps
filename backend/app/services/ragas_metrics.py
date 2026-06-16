from typing import List, Optional
from pydantic import BaseModel

class RAGASMetrics(BaseModel):
    # Core RAGAS triad
    context_relevance: float        # 0.0-1.0: retrieved context relevant to query?
    faithfulness: float             # 0.0-1.0: answer supported by retrieved context?
    answer_relevance: float         # 0.0-1.0: answer addresses the user query?
    
    # Extended metrics
    groundedness: float             # 0.0-1.0: answer dependent on sources, not priors?
    context_recall: Optional[float] # 0.0-1.0: did retrieval find what was needed?
    
    # Metadata
    evaluation_method: str          # "tfidf_deterministic"
    hallucination_risk: str         # "low" | "medium" | "high"
    overall_score: float            # weighted average
    
    # RAGAS-compatible labels for display
    labels: dict                    # human-readable label map

class RAGASEvaluator:
    """
    Maps existing TF-IDF eval scores to RAGAS-compatible metric taxonomy.
    No new LLM calls - purely deterministic remapping of existing scores.
    """
    
    METRIC_WEIGHTS = {
        "context_relevance": 0.25,
        "faithfulness": 0.35,
        "answer_relevance": 0.25,
        "groundedness": 0.15,
    }
    
    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[dict],
        generated_answer: str,
        existing_eval_scores: Optional[dict] = None
    ) -> RAGASMetrics:
        """
        Compute RAGAS-compatible metrics using deterministic TF-IDF scoring.
        If existing_eval_scores provided (from current eval engine), map them.
        Otherwise compute from scratch.
        """
        context_relevance = self._compute_context_relevance(
            query, retrieved_chunks, existing_eval_scores
        )
        faithfulness = self._compute_faithfulness(
            generated_answer, retrieved_chunks, existing_eval_scores
        )
        answer_relevance = self._compute_answer_relevance(
            query, generated_answer, existing_eval_scores
        )
        groundedness = self._compute_groundedness(
            generated_answer, retrieved_chunks, existing_eval_scores
        )
        
        overall = sum(
            score * self.METRIC_WEIGHTS[metric]
            for metric, score in {
                "context_relevance": context_relevance,
                "faithfulness": faithfulness,
                "answer_relevance": answer_relevance,
                "groundedness": groundedness,
            }.items()
        )
        
        hallucination_risk = (
            "low" if faithfulness >= 0.75
            else "medium" if faithfulness >= 0.5
            else "high"
        )
        
        return RAGASMetrics(
            context_relevance=round(context_relevance, 3),
            faithfulness=round(faithfulness, 3),
            answer_relevance=round(answer_relevance, 3),
            groundedness=round(groundedness, 3),
            context_recall=None,  # Requires ground truth, skip for now
            evaluation_method="tfidf_deterministic",
            hallucination_risk=hallucination_risk,
            overall_score=round(overall, 3),
            labels={
                "context_relevance": self._label(context_relevance),
                "faithfulness": self._label(faithfulness),
                "answer_relevance": self._label(answer_relevance),
                "groundedness": self._label(groundedness),
                "overall": self._label(overall),
            }
        )
    
    def _compute_context_relevance(
        self, 
        query: str, 
        chunks: List[dict],
        existing: Optional[dict]
    ) -> float:
        """
        How relevant are retrieved chunks to the query?
        Uses TF-IDF cosine similarity between query and each chunk.
        Returns average similarity across top chunks.
        """
        if existing and "context_relevance" in existing:
            return float(existing["context_relevance"])
        
        if not chunks:
            return 0.0
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        chunk_texts = [c.get("content", "") for c in chunks[:5]]
        if not any(chunk_texts):
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            all_texts = [query] + chunk_texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            query_vec = tfidf_matrix[0]
            chunk_vecs = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vec, chunk_vecs)[0]
            return float(np.mean(similarities))
        except Exception:
            return 0.5
    
    def _compute_faithfulness(
        self, 
        answer: str, 
        chunks: List[dict],
        existing: Optional[dict]
    ) -> float:
        """
        Is every claim in the answer supported by retrieved chunks?
        Reuses existing faithfulness score if available.
        """
        if existing and "faithfulness" in existing:
            return float(existing["faithfulness"])
        if existing and "faithfulness_score" in existing:
            return float(existing["faithfulness_score"])
        if existing and "hallucination_score" in existing:
            return 1.0 - float(existing["hallucination_score"])
        
        if not chunks or not answer:
            return 0.0
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        context = " ".join(c.get("content", "") for c in chunks[:5])
        
        try:
            sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
            if not sentences:
                return 0.5
            
            vectorizer = TfidfVectorizer(stop_words="english")
            all_texts = [context] + sentences
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            context_vec = tfidf_matrix[0]
            sentence_vecs = tfidf_matrix[1:]
            
            import numpy as np
            similarities = cosine_similarity(context_vec, sentence_vecs)[0]
            return float(np.mean(similarities))
        except Exception:
            return 0.5
    
    def _compute_answer_relevance(
        self, 
        query: str, 
        answer: str,
        existing: Optional[dict]
    ) -> float:
        """
        Does the answer address what the user asked?
        TF-IDF similarity between query and answer.
        """
        if existing and "answer_relevance" in existing:
            return float(existing["answer_relevance"])
        
        if not query or not answer:
            return 0.0
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([query, answer])
            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            return float(similarity)
        except Exception:
            return 0.5
    
    def _compute_groundedness(
        self, 
        answer: str, 
        chunks: List[dict],
        existing: Optional[dict]
    ) -> float:
        """
        Is the answer grounded in retrieved sources rather than LLM priors?
        Proxy: token overlap between answer and retrieved context.
        """
        if existing and "groundedness" in existing:
            return float(existing["groundedness"])
        
        if not chunks or not answer:
            return 0.0
        
        context_tokens = set(
            " ".join(c.get("content", "") for c in chunks).lower().split()
        )
        answer_tokens = set(answer.lower().split())
        
        if not answer_tokens:
            return 0.0
        
        # Remove stopwords for meaningful overlap
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in",
                    "of", "to", "and", "or", "for", "with", "by", "at"}
        answer_content_tokens = answer_tokens - stopwords
        
        if not answer_content_tokens:
            return 0.5
        
        overlap = len(answer_content_tokens & context_tokens)
        return min(1.0, overlap / len(answer_content_tokens))
    
    def _label(self, score: float) -> str:
        if score >= 0.8:
            return "excellent"
        elif score >= 0.65:
            return "good"
        elif score >= 0.5:
            return "acceptable"
        elif score >= 0.35:
            return "poor"
        else:
            return "failing"
    
    def to_display_dict(self, metrics: RAGASMetrics) -> dict:
        """Format for frontend display and API response."""
        return {
            "ragas_metrics": {
                "context_relevance": {
                    "score": metrics.context_relevance,
                    "label": metrics.labels["context_relevance"],
                    "description": "How relevant were retrieved chunks to your query"
                },
                "faithfulness": {
                    "score": metrics.faithfulness,
                    "label": metrics.labels["faithfulness"],
                    "description": "How well the answer is supported by retrieved context"
                },
                "answer_relevance": {
                    "score": metrics.answer_relevance,
                    "label": metrics.labels["answer_relevance"],
                    "description": "How directly the answer addresses your question"
                },
                "groundedness": {
                    "score": metrics.groundedness,
                    "label": metrics.labels["groundedness"],
                    "description": "How much the answer relies on retrieved sources vs model knowledge"
                },
                "overall_score": metrics.overall_score,
                "hallucination_risk": metrics.hallucination_risk,
                "evaluation_method": metrics.evaluation_method,
            }
        }
