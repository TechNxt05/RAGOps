from typing import List, Optional
import re

def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ContextualCompressor:
    """
    Query-aware, budget-driven sentence-level compression.

    Context is only compressed when it exceeds the token budget: dropping text
    that fits is pure loss (a lexical filter used to drop answer-bearing
    paraphrases such as "International travel must be approved by a director"
    for "Who signs off on overseas trips?"). When over budget, sentences are
    ranked by embedding similarity to the query (meaning, not shared words),
    factual sentences (numbers, dates, named entities) get a bonus, every chunk
    keeps at least its best sentence, and original order is preserved.
    """

    FACTUAL_BONUS = 0.15

    def __init__(self, similarity_threshold: float = 0.15):
        self.similarity_threshold = similarity_threshold

    def compress_chunks(
        self,
        query: str,
        chunks: List[dict],
        max_total_tokens: int = 2000,
        min_sentences_per_chunk: int = 1
    ) -> dict:
        """
        Fit the chunks into max_total_tokens.

        Returns:
            {
                "compressed_chunks": List[dict],
                "original_token_estimate": int,
                "compressed_token_estimate": int,
                "compression_ratio": float,
                "sentences_kept": int,
                "sentences_dropped": int
            }
        """
        original_tokens = sum(_tokens(c.get("content", "")) for c in chunks)
        if original_tokens <= max_total_tokens:
            return {
                "compressed_chunks": [
                    {**c, "original_content": c.get("content", ""), "compression_applied": False,
                     "sentences_kept": None, "sentences_dropped": 0}
                    for c in chunks
                ],
                "original_token_estimate": original_tokens,
                "compressed_token_estimate": original_tokens,
                "compression_ratio": 0.0,
                "sentences_kept": None,
                "sentences_dropped": 0,
            }

        # Over budget: rank every sentence of every chunk by semantic relevance.
        per_chunk = [self._split_sentences(c.get("content", "")) or [c.get("content", "")] for c in chunks]
        flat = [(ci, si, s) for ci, sents in enumerate(per_chunk) for si, s in enumerate(sents)]
        scores = self._semantic_scores(query, [s for _, _, s in flat])
        ranked = sorted(
            range(len(flat)),
            key=lambda i: scores[i] + (self.FACTUAL_BONUS if self._contains_factual_content(flat[i][2]) else 0.0),
            reverse=True,
        )

        keep = set()
        used = 0
        # 1) the best sentence(s) of every chunk, so no retrieved source vanishes
        for ci in range(len(chunks)):
            best = [i for i in ranked if flat[i][0] == ci][:min_sentences_per_chunk]
            for i in best:
                keep.add(i)
                used += _tokens(flat[i][2])
        # 2) then the globally most relevant sentences until the budget is spent
        for i in ranked:
            if i in keep:
                continue
            cost = _tokens(flat[i][2])
            if used + cost > max_total_tokens:
                continue
            keep.add(i)
            used += cost

        compressed_chunks = []
        total_kept = total_dropped = 0
        for ci, chunk in enumerate(chunks):
            idx = [i for i in range(len(flat)) if flat[i][0] == ci]
            kept_text = " ".join(flat[i][2] for i in idx if i in keep)
            kept = sum(1 for i in idx if i in keep)
            dropped = len(idx) - kept
            total_kept += kept
            total_dropped += dropped
            compressed_chunks.append({
                **chunk,
                "content": kept_text,
                "original_content": chunk.get("content", ""),
                "compression_applied": dropped > 0,
                "sentences_kept": kept,
                "sentences_dropped": dropped,
            })

        compressed_tokens = sum(_tokens(c["content"]) for c in compressed_chunks)
        return {
            "compressed_chunks": compressed_chunks,
            "original_token_estimate": original_tokens,
            "compressed_token_estimate": compressed_tokens,
            "compression_ratio": round(1.0 - compressed_tokens / original_tokens, 3) if original_tokens else 0.0,
            "sentences_kept": total_kept,
            "sentences_dropped": total_dropped,
        }

    def _semantic_scores(self, query: str, sentences: List[str]) -> List[float]:
        try:
            from app.services import model_registry
            q = model_registry.embed_query(query)
            vecs = model_registry.embed_documents(sentences)
            return [model_registry.cosine(q, v) for v in vecs]
        except Exception:
            return [self._sentence_relevance_score(query, s) for s in sentences]
    
    def _compress_text(
        self, 
        query: str, 
        text: str,
        min_sentences: int
    ) -> tuple:
        """
        Extract query-relevant sentences from text.
        Always preserve sentences containing entities/numbers/dates.
        Returns (compressed_text, kept_count, dropped_count)
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= min_sentences:
            return text, len(sentences), 0
        
        # Score each sentence for relevance to query
        scored = []
        for sent in sentences:
            score = self._sentence_relevance_score(query, sent)
            is_factual = self._contains_factual_content(sent)
            scored.append({
                "text": sent,
                "score": score,
                "is_factual": is_factual,
                # Always keep factual sentences regardless of score
                "keep": score >= self.similarity_threshold or is_factual
            })
        
        # Ensure minimum sentences kept
        kept_sentences = [s for s in scored if s["keep"]]
        if len(kept_sentences) < min_sentences:
            # Keep top N by score even if below threshold
            sorted_by_score = sorted(scored, key=lambda x: x["score"], reverse=True)
            for s in sorted_by_score[:min_sentences]:
                s["keep"] = True
            kept_sentences = [s for s in scored if s["keep"]]
        
        # Preserve original order
        kept_texts = [s["text"] for s in scored if s["keep"]]
        dropped_texts = [s["text"] for s in scored if not s["keep"]]
        
        compressed = " ".join(kept_texts)
        return compressed, len(kept_texts), len(dropped_texts)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling common edge cases."""
        # Handle abbreviations and decimal numbers before splitting
        text = re.sub(r"(\d+)\.(\d+)", r"\1[DOT]\2", text)
        text = re.sub(r"\b(Mr|Mrs|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g)\.", r"\1[DOT]", text)
        
        parts = re.split(r"(?<=[.!?])\s+", text)
        parts = [s.replace("[DOT]", ".").strip() for s in parts if s.strip()]

        # Merge very short fragments ("Price: $40.", headings) into the next
        # sentence instead of discarding them: they often carry the answer.
        sentences, pending = [], ""
        for part in parts:
            if len(part.split()) < 4:
                pending = f"{pending} {part}".strip()
                continue
            sentences.append(f"{pending} {part}".strip() if pending else part)
            pending = ""
        if pending:
            if sentences:
                sentences[-1] = f"{sentences[-1]} {pending}"
            else:
                sentences.append(pending)
        return sentences
    
    def _sentence_relevance_score(self, query: str, sentence: str) -> float:
        """
        Compute TF-IDF cosine similarity between query and sentence.
        """
        try:
            from app.utils.tfidf import cosine, tfidf_vectors

            q_vec, s_vec = tfidf_vectors([query, sentence], stop_words="english")
            return cosine(q_vec, s_vec)
        except Exception:
            # Fallback: simple keyword overlap
            query_words = set(query.lower().split())
            sent_words = set(sentence.lower().split())
            overlap = len(query_words & sent_words)
            return overlap / max(len(query_words), 1)
    
    def _contains_factual_content(self, sentence: str) -> bool:
        """
        Detect sentences containing critical facts that must be preserved:
        numbers, dates, percentages, specific entities, money amounts.
        """
        case_insensitive_patterns = [
            r"\b\d+(?:\.\d+)?%",                            # percentages e.g. 99.5%
            r"\$[\d,]+(?:\.\d{2})?",                        # money e.g. $4.5 million
            r"\b\d+\s*(?:microseconds|ms|ns|seconds|qubits|gb|mb|ghz|mhz|m/s|km)\b", # physical units
            r"\b(?:19|20)\d{2}\b",                          # specific 4-digit years
            r"\b\d+\s+(?:days|months|years|hours|weeks)\b",  # durations
            r"\b(?:section|clause|article|table|figure)\s+[\d\.]+", # references
        ]
        
        for pattern in case_insensitive_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True

        # Proper nouns (e.g. "Google Sycamore", "Peter Shor") - strictly case sensitive
        # Skip checking the very first word of the sentence to avoid matching capitalized sentence starters
        words = sentence.split()
        if len(words) >= 2:
            body_text = " ".join(words[1:])
            if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", body_text):
                return True
                
        return False
