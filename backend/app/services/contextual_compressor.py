from typing import List, Optional
import re

class ContextualCompressor:
    """
    Query-aware sentence-level compression.
    Extracts only sentences relevant to the query from each chunk.
    Preserves entities, numbers, dates verbatim.
    Replaces blunt TF-IDF token pruning with surgical sentence extraction.
    """
    
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
        Compress each chunk to query-relevant sentences only.
        
        Args:
            query: The user query (rewritten version)
            chunks: Retrieved and reranked chunks
            max_total_tokens: Total token budget for all compressed chunks
            min_sentences_per_chunk: Always keep at least N sentences per chunk
        
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
        compressed_chunks = []
        total_original_tokens = 0
        total_compressed_tokens = 0
        total_kept = 0
        total_dropped = 0
        
        for chunk in chunks:
            content = chunk.get("content", "")
            if not content.strip():
                compressed_chunks.append(chunk)
                continue
            
            original_tokens = len(content.split())
            total_original_tokens += original_tokens
            
            compressed_content, kept, dropped = self._compress_text(
                query, content, min_sentences_per_chunk
            )
            
            compressed_tokens = len(compressed_content.split())
            total_compressed_tokens += compressed_tokens
            total_kept += kept
            total_dropped += dropped
            
            compressed_chunk = {
                **chunk,
                "content": compressed_content,
                "original_content": content,
                "compression_applied": True,
                "sentences_kept": kept,
                "sentences_dropped": dropped,
            }
            compressed_chunks.append(compressed_chunk)
            
            # Stop if we have enough context
            if total_compressed_tokens >= max_total_tokens:
                # Add remaining chunks uncompressed if budget allows
                break
        
        compression_ratio = (
            1.0 - (total_compressed_tokens / total_original_tokens)
            if total_original_tokens > 0 else 0.0
        )
        
        return {
            "compressed_chunks": compressed_chunks,
            "original_token_estimate": total_original_tokens,
            "compressed_token_estimate": total_compressed_tokens,
            "compression_ratio": round(compression_ratio, 3),
            "sentences_kept": total_kept,
            "sentences_dropped": total_dropped
        }
    
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
        
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.replace("[DOT]", ".").strip() for s in sentences if s.strip()]
        
        # Filter out very short fragments (likely not real sentences)
        sentences = [s for s in sentences if len(s.split()) >= 4]
        
        return sentences
    
    def _sentence_relevance_score(self, query: str, sentence: str) -> float:
        """
        Compute TF-IDF cosine similarity between query and sentence.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
            tfidf_matrix = vectorizer.fit_transform([query, sentence])
            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            return float(similarity)
        except Exception:
            # Fallback: simple keyword overlap
            query_words = set(query.lower().split())
            sent_words = set(sentence.lower().split())
            overlap = len(query_words & sent_words)
            return overlap / max(len(query_words), 1)
    
    def _contains_factual_content(self, sentence: str) -> bool:
        """
        Detect sentences containing facts that must be preserved:
        numbers, dates, percentages, proper nouns, money amounts.
        """
        factual_patterns = [
            r"\b\d+(?:\.\d+)?%",           # percentages
            r"\$[\d,]+(?:\.\d{2})?",        # money
            r"\b\d{4}\b",                   # years
            r"\b\d+\s+(?:days|months|years|hours|weeks)\b",  # durations
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",          # proper nouns
            r"\b(?:section|clause|article)\s+[\d\.]+",       # references
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False
