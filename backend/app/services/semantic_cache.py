"""
Persistent semantic answer cache.

A hit requires BOTH:
  1. embedding cosine similarity >= threshold (default 0.92), and
  2. the same set of content keywords (stopwords removed, light plural folding).

(2) is what makes the cache safe: "What is the capital of France?" and
"What is the capital of Spain?" are ~0.9 similar as sentences, but their keyword
sets differ, so the second can never be answered with the first's response.
Entries are scoped to (project, kb_version), so any document change invalidates
them. Stored in Postgres so they survive restarts.
"""

import re
from datetime import datetime, timedelta
from typing import List, Optional, Set

from sqlmodel import Session, delete, select

from app.models.rag import SemanticCacheEntry
from app.services import model_registry

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "in", "on", "at",
    "to", "for", "from", "by", "with", "about", "as", "into", "and", "or", "but", "if", "then",
    "what", "whats", "which", "who", "whom", "whose", "when", "where", "why", "how", "do", "does",
    "did", "can", "could", "would", "should", "will", "shall", "may", "might", "must", "i", "me",
    "my", "we", "our", "you", "your", "it", "its", "this", "that", "these", "those", "there",
    "please", "tell", "explain", "describe", "give", "show", "list", "some", "any", "all", "s",
}
MAX_ENTRIES_SCANNED = 500
TTL = timedelta(days=7)


def keywords(text: str) -> List[str]:
    out: Set[str] = set()
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.add(tok)
    return sorted(out)


def lookup(session: Session, project_id: int, kb_version: int, query: str, threshold: float = 0.92,
           model: Optional[str] = None) -> Optional[SemanticCacheEntry]:
    kw = keywords(query)
    if not kw:
        return None
    entries = session.exec(
        select(SemanticCacheEntry)
        .where(SemanticCacheEntry.project_id == project_id)
        .where(SemanticCacheEntry.kb_version == kb_version)
        .where(SemanticCacheEntry.created_at >= datetime.utcnow() - TTL)
        .where(SemanticCacheEntry.model == model if model else True)
        .order_by(SemanticCacheEntry.id.desc())
        .limit(MAX_ENTRIES_SCANNED)
    ).all()
    candidates = [e for e in entries if list(e.keywords or []) == kw]
    if not candidates:
        return None
    q_vec = model_registry.embed_query(query)
    best, best_sim = None, 0.0
    for entry in candidates:
        sim = model_registry.cosine(q_vec, entry.embedding)
        if sim > best_sim:
            best, best_sim = entry, sim
    if best is None or best_sim < threshold:
        return None
    best.hits += 1
    session.add(best)
    session.commit()
    return best


def store(session: Session, project_id: int, kb_version: int, query: str, model: str, response: dict) -> None:
    kw = keywords(query)
    if not kw:
        return
    session.add(SemanticCacheEntry(
        project_id=project_id,
        kb_version=kb_version,
        model=model,
        query=query,
        keywords=kw,
        embedding=model_registry.embed_query(query),
        response=response,
    ))
    # Drop entries for superseded knowledge-base versions.
    session.exec(
        delete(SemanticCacheEntry)
        .where(SemanticCacheEntry.project_id == project_id)
        .where(SemanticCacheEntry.kb_version < kb_version)
    )
    session.commit()


class InMemorySemanticCache:
    """Same matching rules as the persistent cache (keywords must match AND
    cosine >= threshold), kept in memory. Used by the benchmark runner so cached
    variants can never be scored on another question's answer."""

    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self._entries: dict = {}

    def get(self, query: str, project_id: int, kb_version: int) -> Optional[str]:
        kw = tuple(keywords(query))
        bucket = self._entries.get((project_id, kb_version, kw))
        if not kw or not bucket:
            return None
        q_vec = model_registry.embed_query(query)
        best = max(bucket, key=lambda e: model_registry.cosine(q_vec, e[0]))
        return best[1] if model_registry.cosine(q_vec, best[0]) >= self.threshold else None

    def set(self, query: str, response: str, project_id: int, kb_version: int) -> None:
        kw = tuple(keywords(query))
        if kw:
            self._entries.setdefault((project_id, kb_version, kw), []).append((model_registry.embed_query(query), response))
