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
        Uses word overlap + sub-phrase TF-IDF cosine similarity.
        """
        if existing and "context_relevance" in existing:
            return float(existing["context_relevance"])
        
        if not chunks:
            return 0.0
        
        chunk_texts = [c.get("content", "") for c in chunks[:5] if c.get("content")]
        if not chunk_texts:
            return 0.0
        
        try:
            import re
            
            # Text normalization function
            def normalize(text):
                # Remove punctuation and lowercase
                text = re.sub(r'[^\w\s]', '', text).lower()
                words = set(text.split())
                # Remove common stop words to avoid artificial inflation
                stop_words = {"what", "is", "the", "a", "an", "of", "in", "to", "and", "or", "for", "on", "with"}
                return words - stop_words

            q_words = normalize(query)
            if not q_words:
                return 0.0
            
            overlap_scores = []
            for ct in chunk_texts:
                c_words = normalize(ct)
                if not c_words:
                    overlap_scores.append(0.0)
                    continue
                # Jaccard similarity
                intersection = len(q_words & c_words)
                union = len(q_words | c_words)
                overlap = intersection / union if union > 0 else 0.0
                # Scale up a bit since Jaccard between a short query and long chunk is naturally very small
                # We care more about query coverage (recall) than precision in the chunk
                query_coverage = intersection / len(q_words)
                # Combine Jaccard and Query Coverage
                score = (0.3 * overlap) + (0.7 * query_coverage)
                overlap_scores.append(score)
            
            # Context relevance is highest score among top chunks
            combined = max(overlap_scores) if overlap_scores else 0.0
            return float(min(1.0, max(0.0, combined)))
        except Exception as e:
            return 0.0
    
    def _compute_faithfulness(
        self, 
        answer: str, 
        chunks: List[dict],
        existing: Optional[dict]
    ) -> float:
        """
        Is every claim in the answer supported by retrieved chunks?
        Uses sentence-level grounding: for each answer sentence, check if
        key content words appear in the context.
        """
        if existing and "faithfulness" in existing:
            return float(existing["faithfulness"])
        if existing and "faithfulness_score" in existing:
            return float(existing["faithfulness_score"])
        if existing and "hallucination_score" in existing:
            return 1.0 - float(existing["hallucination_score"])
        
        if not chunks or not answer or not answer.strip():
            return 0.0
        
        import re
        context_text = " ".join(c.get("content", "") for c in chunks[:5]).lower()
        context_text = re.sub(r'[^\w\s]', '', context_text)
        if not context_text.strip():
            return 0.0
        
        context_words = set(context_text.split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "shall", "can",
                      "to", "of", "in", "for", "on", "with", "at", "by", "from",
                      "and", "or", "but", "not", "no", "so", "if", "as", "it",
                      "its", "this", "that", "these", "those", "than", "then"}
        
        # Split answer into sentences
        sentences = re.split(r'[.!?\n]', answer)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        
        if not sentences:
            return 0.0
        
        sentence_scores = []
        for sent in sentences:
            sent_clean = re.sub(r'[^\w\s]', '', sent.lower())
            sent_words = [w for w in sent_clean.split() if w not in stop_words and len(w) > 2]
            if not sent_words:
                sentence_scores.append(1.0)  # Trivial sentence, not a claim
                continue
            matched = sum(1 for w in sent_words if w in context_words)
            sentence_scores.append(matched / len(sent_words))
        
        return float(min(1.0, max(0.0, sum(sentence_scores) / len(sentence_scores))))
    
    def _compute_answer_relevance(
        self, 
        query: str, 
        answer: str,
        existing: Optional[dict]
    ) -> float:
        """
        Does the answer address what the user asked?
        Combines query keyword coverage + TF-IDF cosine similarity.
        """
        if existing and "answer_relevance" in existing:
            return float(existing["answer_relevance"])
        
        if not query or not answer or not answer.strip():
            return 0.0
        
        try:
            import re
            stop_words = {"what", "is", "the", "a", "an", "of", "in", "to", "and",
                          "or", "for", "on", "with", "how", "does", "do", "are",
                          "can", "you", "this", "that", "was", "were"}
            
            def clean(text):
                return set(w for w in re.sub(r'[^\w\s]', '', text.lower()).split()
                          if w not in stop_words and len(w) > 1)
            
            q_words = clean(query)
            a_words = clean(answer)
            if not q_words or not a_words:
                return 0.0
            
            # Query coverage: how many query keywords appear in the answer
            query_coverage = len(q_words & a_words) / len(q_words)
            
            # TF-IDF cosine similarity
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            tfidf_matrix = vectorizer.fit_transform([query, answer])
            tfidf_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            
            # Bonus for longer, more detailed answers (up to a cap)
            answer_length_bonus = min(0.1, len(a_words) / 200)
            
            score = (0.35 * query_coverage) + (0.55 * tfidf_sim) + answer_length_bonus
            return float(min(1.0, max(0.0, score)))
        except Exception:
            return 0.0
    
    def _compute_groundedness(
        self, 
        answer: str, 
        chunks: List[dict],
        existing: Optional[dict]
    ) -> float:
        """
        Is the answer grounded in retrieved sources rather than LLM priors?
        Uses bigram overlap to catch phrase-level grounding, not just word-level.
        """
        if existing and "groundedness" in existing:
            return float(existing["groundedness"])
        
        if not chunks or not answer or not answer.strip():
            return 0.0
        
        import re
        context_text = re.sub(r'[^\w\s]', '', " ".join(c.get("content", "") for c in chunks).lower())
        context_tokens = context_text.split()
        answer_text = re.sub(r'[^\w\s]', '', answer.lower())
        answer_tokens = [w for w in answer_text.split() if len(w) > 2]
        
        if not answer_tokens:
            return 0.0
        
        # Unigram grounding
        context_set = set(context_tokens)
        stop_words = {"the", "and", "for", "are", "but", "not", "you", "all",
                      "can", "had", "her", "was", "one", "our", "out", "has",
                      "with", "this", "that", "from", "they", "been", "have",
                      "its", "will", "would", "there", "their"}
        content_tokens = [w for w in answer_tokens if w not in stop_words]
        if not content_tokens:
            return 1.0  # Only stop words, trivially grounded
        
        unigram_grounded = sum(1 for t in content_tokens if t in context_set)
        unigram_score = unigram_grounded / len(content_tokens)
        
        # Bigram grounding for phrase-level accuracy
        context_bigrams = set(zip(context_tokens, context_tokens[1:]))
        answer_bigrams = list(zip(answer_tokens, answer_tokens[1:]))
        if answer_bigrams:
            bigram_grounded = sum(1 for bg in answer_bigrams if bg in context_bigrams)
            bigram_score = bigram_grounded / len(answer_bigrams)
        else:
            bigram_score = unigram_score
        
        score = (0.6 * unigram_score) + (0.4 * bigram_score)
        return float(min(1.0, max(0.0, score)))
    
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
