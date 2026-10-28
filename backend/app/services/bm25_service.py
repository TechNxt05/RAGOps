import re
import threading
from typing import Dict, List, Optional, Set, Tuple

from sqlmodel import Session, select

from app.models.rag import Chunk, Document


class _ProjectIndex:
    __slots__ = ("bm25", "chunk_ids", "doc_ids", "sources")

    def __init__(self, bm25, chunk_ids: List[int], doc_ids: List[int], sources: List[str]):
        self.bm25 = bm25
        self.chunk_ids = chunk_ids
        self.doc_ids = doc_ids
        self.sources = sources


class BM25IndexManager:
    """
    Per-project BM25 (Okapi) index built from the chunk table.

    Indexes are cached in memory keyed by (project_id, kb_version). Every write to
    a project's knowledge base bumps kb_version, so stale indexes are never served
    and there is no pickle file on disk to go missing on an ephemeral filesystem.
    """

    def __init__(self):
        self._cache: Dict[int, Tuple[int, Optional[_ProjectIndex]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenizer — lowercase, split on non-alphanumeric."""
        return re.findall(r'\b\w+\b', text.lower())

    def _build(self, session: Session, project_id: int) -> Optional[_ProjectIndex]:
        from rank_bm25 import BM25Okapi
        rows = session.exec(
            select(Chunk.id, Chunk.content, Document.id, Document.filename)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.project_id == project_id)
            .where(Document.is_active == True)  # noqa: E712
            .where(Document.processed == True)  # noqa: E712
            .order_by(Chunk.id)
        ).all()
        if not rows:
            return None
        bm25 = BM25Okapi([self._tokenize(r[1]) for r in rows])
        return _ProjectIndex(bm25, [r[0] for r in rows], [r[2] for r in rows], [r[3] for r in rows])

    def get(self, session: Session, project_id: int, kb_version: int) -> Optional[_ProjectIndex]:
        cached = self._cache.get(project_id)
        if cached and cached[0] == kb_version:
            return cached[1]
        with self._lock:
            cached = self._cache.get(project_id)
            if cached and cached[0] == kb_version:
                return cached[1]
            index = self._build(session, project_id)
            self._cache[project_id] = (kb_version, index)
            return index

    def search(
        self,
        session: Session,
        project_id: int,
        kb_version: int,
        query: str,
        top_k: int = 20,
        document_id: Optional[int] = None,
        sources: Optional[Set[str]] = None,
    ) -> List[Tuple[int, float]]:
        """Return [(chunk_id, bm25_score)] best-first. Chunks with no term overlap
        (score 0) are not returned: they carry no lexical evidence."""
        index = self.get(session, project_id, kb_version)
        if index is None:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = index.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: List[Tuple[int, float]] = []
        for i in ranked:
            if scores[i] <= 0:
                break
            if document_id is not None and index.doc_ids[i] != document_id:
                continue
            if sources is not None and index.sources[i] not in sources:
                continue
            out.append((index.chunk_ids[i], float(scores[i])))
            if len(out) >= top_k:
                break
        return out

    def invalidate(self, project_id: int) -> None:
        self._cache.pop(project_id, None)

    # Backwards-compatible no-ops for callers of the old pickle-based API.
    def delete_index(self, project_id) -> None:
        self.invalidate(int(project_id))


# Singleton instance
bm25_manager = BM25IndexManager()
