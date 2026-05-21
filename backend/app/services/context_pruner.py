from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Any

class ContextPruner:
    """
    Prunes irrelevant retrieved chunks using a fast TF-IDF cosine similarity check.
    Calculates similarity between the query and each chunk.
    Only chunks exceeding a threshold are kept.
    """
    
    def prune(
        self,
        query: str,
        chunks: List[Any],  # list of chunks/documents
        threshold: float = 0.1
    ) -> Tuple[List[Any], int, int, float]:
        """
        Prunes chunks below the threshold.
        Returns:
            - list of pruned chunks
            - original chunk count
            - pruned chunk count
            - reduction percentage
        """
        if not chunks:
            return [], 0, 0, 0.0
            
        if threshold <= 0.0:
            return chunks, len(chunks), len(chunks), 0.0
            
        # Extract page_content
        texts = []
        for c in chunks:
            if hasattr(c, 'page_content'):
                texts.append(c.page_content)
            elif isinstance(c, tuple) and len(c) > 0 and hasattr(c[0], 'page_content'):
                texts.append(c[0].page_content)
            elif isinstance(c, dict) and 'content' in c:
                texts.append(c['content'])
            else:
                texts.append(str(c))
                
        # Build TF-IDF
        try:
            # Simple vectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            # Fit on corpus (chunks + query)
            tfidf_matrix = vectorizer.fit_transform(texts + [query])
            
            # Last row is the query
            query_vector = tfidf_matrix[-1]
            # Other rows are the chunks
            chunk_vectors = tfidf_matrix[:-1]
            
            # Calculate cosine similarities
            similarities = cosine_similarity(chunk_vectors, query_vector).flatten()
            
            pruned_chunks = []
            for idx, score in enumerate(similarities):
                if score >= threshold:
                    pruned_chunks.append(chunks[idx])
            
            # Edge case: if we pruned EVERYTHING, keep at least the top 1 chunk to avoid empty context
            if not pruned_chunks and chunks:
                sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
                pruned_chunks.append(chunks[sorted_indices[0]])
                
            orig_len = len(chunks)
            pruned_len = len(pruned_chunks)
            reduction_pct = ((orig_len - pruned_len) / orig_len) * 100.0 if orig_len > 0 else 0.0
            
            return pruned_chunks, orig_len, pruned_len, reduction_pct
            
        except Exception as e:
            # Graceful fallback: return everything if vectorization/fit fails (e.g. empty vocabulary)
            print(f"Error in context pruning: {e}")
            return chunks, len(chunks), len(chunks), 0.0

context_pruner = ContextPruner()
