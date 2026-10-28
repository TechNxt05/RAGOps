"""
pgvector-backed vector store (with a pure-Python fallback for SQLite dev/tests).

Replaces the single on-disk FAISS index that was shared by every project.
Vectors are rows in `chunkembedding`, so:
  * each project is filtered in SQL (no cross-project crowding or overwrites),
  * inserts/deletes share the transaction of the chunk rows they belong to,
  * nothing lives on the container's ephemeral disk.

Search is exact (sequential cosine scan) for projects up to EXACT_SEARCH_LIMIT
vectors, matching FAISS IndexFlatL2's exact recall; larger projects use a
per-dimension partial HNSW index.
"""

import logging
import threading
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from sqlalchemy import text
from sqlmodel import Session, delete, func, select

from app.db import IS_POSTGRES, engine
from app.models.rag import ChunkEmbedding, Document
from app.services.model_registry import cosine

logger = logging.getLogger(__name__)

EXACT_SEARCH_LIMIT = 50_000
HNSW_MAX_DIM = 2000  # pgvector's limit for indexing the `vector` type

_indexed_dims: Set[int] = set()
_index_lock = threading.Lock()


def _vec_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vec) + "]"


def ensure_hnsw_index(dim: int) -> None:
    if not IS_POSTGRES or dim > HNSW_MAX_DIM or dim in _indexed_dims:
        return
    with _index_lock:
        if dim in _indexed_dims:
            return
        dim = int(dim)
        with engine.begin() as conn:
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS ix_chunkembedding_hnsw_{dim} ON chunkembedding "
                f"USING hnsw ((embedding::vector({dim})) vector_cosine_ops) WHERE dim = {dim}"
            ))
        _indexed_dims.add(dim)


def add_embeddings(
    session: Session,
    rows: Iterable[Tuple[int, int, int]],  # (chunk_id, document_id, project_id)
    vectors: List[List[float]],
    model: str,
) -> int:
    """Stage embedding rows in the caller's transaction (caller commits)."""
    count = 0
    for (chunk_id, document_id, project_id), vec in zip(rows, vectors):
        session.add(ChunkEmbedding(
            chunk_id=chunk_id,
            document_id=document_id,
            project_id=project_id,
            model=model,
            dim=len(vec),
            embedding=list(vec),
        ))
        count += 1
    return count


def delete_for_chunks(session: Session, chunk_ids: List[int]) -> None:
    if chunk_ids:
        session.exec(delete(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(chunk_ids)))


def delete_for_document(session: Session, document_id: int) -> None:
    session.exec(delete(ChunkEmbedding).where(ChunkEmbedding.document_id == document_id))


def delete_other_models(session: Session, project_id: int, keep_model: str) -> None:
    session.exec(
        delete(ChunkEmbedding)
        .where(ChunkEmbedding.project_id == project_id)
        .where(ChunkEmbedding.model != keep_model)
    )


def embedded_chunk_ids(session: Session, chunk_ids: List[int], model: str) -> Set[int]:
    if not chunk_ids:
        return set()
    rows = session.exec(
        select(ChunkEmbedding.chunk_id)
        .where(ChunkEmbedding.chunk_id.in_(chunk_ids))
        .where(ChunkEmbedding.model == model)
    ).all()
    return set(rows)


def stored_vectors(chunk_ids: List[int], model: str) -> dict:
    """{chunk_id: vector} for chunks already embedded with `model`."""
    if not chunk_ids:
        return {}
    with Session(engine) as s:
        rows = s.exec(
            select(ChunkEmbedding.chunk_id, ChunkEmbedding.embedding)
            .where(ChunkEmbedding.chunk_id.in_(chunk_ids))
            .where(ChunkEmbedding.model == model)
        ).all()
    return {int(cid): [float(x) for x in vec] for cid, vec in rows}


def count_vectors(project_id: int, model: str) -> int:
    with Session(engine) as s:
        return s.exec(
            select(func.count(ChunkEmbedding.id))
            .where(ChunkEmbedding.project_id == project_id)
            .where(ChunkEmbedding.model == model)
        ).one()


def search(
    project_id: int,
    query_vec: List[float],
    model: str,
    k: int,
    document_id: Optional[int] = None,
    sources: Optional[Set[str]] = None,
    exact: bool = True,
) -> List[Tuple[int, float]]:
    """Return [(chunk_id, cosine_similarity)] best-first, active documents only."""
    if k <= 0:
        return []
    if IS_POSTGRES:
        return _search_pg(project_id, query_vec, model, k, document_id, sources, exact)
    return _search_python(project_id, query_vec, model, k, document_id, sources)


def _search_pg(project_id, query_vec, model, k, document_id, sources, exact) -> List[Tuple[int, float]]:
    dim = int(len(query_vec))
    filters = ""
    params = {"p": project_id, "m": model, "q": _vec_literal(query_vec), "k": int(k)}
    if document_id is not None:
        filters += " AND ce.document_id = :doc"
        params["doc"] = int(document_id)
    if sources:
        filters += " AND d.filename = ANY(:sources)"
        params["sources"] = list(sources)
    sql = (
        f"SELECT ce.chunk_id, 1 - (ce.embedding::vector({dim}) <=> CAST(:q AS vector({dim}))) AS sim "
        f"FROM chunkembedding ce JOIN document d ON d.id = ce.document_id "
        f"WHERE ce.project_id = :p AND ce.model = :m AND ce.dim = {dim} "
        f"AND d.is_active = TRUE AND d.processed = TRUE{filters} "
        f"ORDER BY ce.embedding::vector({dim}) <=> CAST(:q AS vector({dim})) LIMIT :k"
    )
    with engine.connect() as conn:
        with conn.begin():
            if exact:
                # Plain index scans off => planner cannot pick the approximate HNSW
                # index; bitmap scans on project_id are still allowed.
                conn.execute(text("SET LOCAL enable_indexscan = off"))
            else:
                conn.execute(text(f"SET LOCAL hnsw.ef_search = {max(100, min(1000, k * 4))}"))
            rows = conn.execute(text(sql), params).fetchall()
    return [(int(r[0]), float(r[1])) for r in rows]


def _search_python(project_id, query_vec, model, k, document_id, sources) -> List[Tuple[int, float]]:
    with Session(engine) as s:
        stmt = (
            select(ChunkEmbedding.chunk_id, ChunkEmbedding.embedding, Document.filename)
            .join(Document, Document.id == ChunkEmbedding.document_id)
            .where(ChunkEmbedding.project_id == project_id)
            .where(ChunkEmbedding.model == model)
            .where(Document.is_active == True)  # noqa: E712
            .where(Document.processed == True)  # noqa: E712
        )
        if document_id is not None:
            stmt = stmt.where(ChunkEmbedding.document_id == document_id)
        rows = s.exec(stmt).all()
    scored = [
        (chunk_id, cosine(query_vec, emb))
        for chunk_id, emb, filename in rows
        if not sources or filename in sources
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
