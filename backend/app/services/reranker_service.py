import os
import sys
from typing import List, Tuple, Any

class BGEReranker:
    """
    Reranks documents/chunks using the BAAI/bge-reranker-base cross-encoder model.
    Falls back gracefully to returning documents as-is if FlagEmbedding is not installed
    or if model initialization/inference fails.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.reranker = None
        self.enabled = False
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        
        self._initialized = True
        try:
            # FlagReranker uses FlagEmbedding package
            from FlagEmbedding import FlagReranker
            print(f"Initializing BGE Reranker with model: {self.model_name}")
            self.reranker = FlagReranker(self.model_name, use_fp16=False)
            self.enabled = True
            print("BGE Reranker initialized successfully.")
        except ImportError:
            print("FlagEmbedding not installed. BGE Reranker is disabled (graceful fallback).")
            self.enabled = False
        except Exception as e:
            print(f"Error initializing BGE Reranker: {e}. Graceful fallback enabled.")
            self.enabled = False

    def rerank(
        self,
        query: str,
        chunks: List[Any],
        top_k: int = 5
    ) -> List[Any]:
        """
        Reranks the chunks against the query and returns the top_k chunks.
        Supports both raw LCDocument objects, strings, tuples, etc.
        """
        if not chunks:
            return []

        # Initialize lazily to avoid loading times on startup if not used
        self.initialize()

        if not self.enabled or self.reranker is None:
            # Fallback: return chunks as-is up to top_k
            return chunks[:top_k]

        try:
            # Extract texts
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

            # Pairs for cross-encoder
            pairs = [[query, text] for text in texts]
            
            # Predict scores
            scores = self.reranker.compute_score(pairs)
            
            # If scores is a single float (only one chunk), turn it into list
            if isinstance(scores, (float, int)):
                scores = [scores]

            # Pair chunks with scores
            ranked_pairs = list(zip(chunks, scores))
            # Sort by score descending
            ranked_pairs.sort(key=lambda x: x[1], reverse=True)

            # Return only the chunks, up to top_k
            return [chunk for chunk, score in ranked_pairs[:top_k]]

        except Exception as e:
            print(f"Error in reranking: {e}. Falling back to default order.")
            return chunks[:top_k]

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
