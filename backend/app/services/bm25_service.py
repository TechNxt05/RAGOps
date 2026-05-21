import pickle
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

class BM25IndexManager:
    """
    Manages per-project BM25 indexes alongside existing FAISS indexes.
    Each project has its own BM25 index stored as a pickle file.
    Mirrors the FAISS per-project isolation pattern.
    """

    def __init__(self, index_dir: str = "faiss_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self._cache: dict[str, any] = {}
        self._corpus_cache: dict[str, list[str]] = {}

    def _index_path(self, project_id: str) -> Path:
        return self.index_dir / f"bm25_{project_id}.pkl"

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer — lowercase, split on non-alphanumeric."""
        return re.findall(r'\b\w+\b', text.lower())

    def build_index(self, project_id: str, chunks: list[str]) -> None:
        """
        Build BM25 index from document chunks.
        Called after FAISS index is built — same trigger point.
        """
        from rank_bm25 import BM25Okapi
        tokenized = [self._tokenize(chunk) for chunk in chunks]
        bm25 = BM25Okapi(tokenized)

        # Save to disk
        with open(self._index_path(project_id), 'wb') as f:
            pickle.dump({"bm25": bm25, "corpus": chunks}, f)

        # Update cache
        self._cache[project_id] = bm25
        self._corpus_cache[project_id] = chunks

    def load_index(self, project_id: str) -> Optional[Tuple[any, list[str]]]:
        """Load BM25 index from disk (with in-memory cache)."""
        if project_id in self._cache:
            return self._cache[project_id], self._corpus_cache[project_id]

        path = self._index_path(project_id)
        if not path.exists():
            return None

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._cache[project_id] = data["bm25"]
            self._corpus_cache[project_id] = data["corpus"]
            return data["bm25"], data["corpus"]
        except Exception:
            return None

    def search(
        self,
        project_id: str,
        query: str,
        top_k: int = 20
    ) -> list[Tuple[str, float]]:
        """
        BM25 search. Returns list of (chunk_text, bm25_score).
        Returns empty list if no index exists for project.
        """
        result = self.load_index(project_id)
        if result is None:
            return []

        bm25, corpus = result
        tokenized_query = self._tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # Get top_k indices by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [(corpus[i], float(scores[i])) for i in top_indices]

    def delete_index(self, project_id: str) -> None:
        """Delete BM25 index when project is deleted or re-chunked."""
        path = self._index_path(project_id)
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        self._cache.pop(project_id, None)
        self._corpus_cache.pop(project_id, None)

    def index_exists(self, project_id: str) -> bool:
        return self._index_path(project_id).exists()


# Singleton instance
bm25_manager = BM25IndexManager()
