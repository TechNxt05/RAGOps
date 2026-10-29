"""
Postgres-backed job queue + in-process worker threads.

Uploads, re-chunking and re-indexing are enqueued as rows in `ingestionjob` and
processed by worker threads started with the API. Because jobs are durable rows,
a restart (deploy, crash, free-tier sleep) never loses queued work, and jobs left
'running' by a crash are re-queued on startup. On Postgres, jobs are claimed with
FOR UPDATE SKIP LOCKED so several workers never pick the same job.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from sqlalchemy import text
from sqlmodel import Session, select

from app.core import llm_keys
from app.core.config import settings
from app.db import IS_POSTGRES, engine
from app.models.rag import IngestionJob

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
STALE_AFTER = timedelta(minutes=30)

_wake = threading.Event()
_stop = threading.Event()
_threads: List[threading.Thread] = []
_handlers: Dict[str, Callable[[Session, IngestionJob], None]] = {}


def handler(kind: str):
    def register(fn):
        _handlers[kind] = fn
        return fn
    return register


def enqueue(
    session: Session,
    kind: str,
    project_id: Optional[int] = None,
    document_id: Optional[int] = None,
    payload: Optional[dict] = None,
    file_bytes: Optional[bytes] = None,
    commit: bool = True,
) -> IngestionJob:
    job = IngestionJob(kind=kind, project_id=project_id, document_id=document_id, payload=payload, file_bytes=file_bytes)
    session.add(job)
    if commit:
        session.commit()
        session.refresh(job)
    _wake.set()
    return job


def _claim_next() -> Optional[int]:
    now = datetime.utcnow()
    with engine.begin() as conn:
        if IS_POSTGRES:
            row = conn.execute(text(
                "UPDATE ingestionjob SET status = 'running', attempts = attempts + 1, updated_at = :now "
                "WHERE id = (SELECT id FROM ingestionjob WHERE status = 'queued' "
                "ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING id"
            ), {"now": now}).first()
            return int(row[0]) if row else None
        row = conn.execute(text("SELECT id FROM ingestionjob WHERE status = 'queued' ORDER BY id LIMIT 1")).first()
        if not row:
            return None
        updated = conn.execute(text(
            "UPDATE ingestionjob SET status = 'running', attempts = attempts + 1, updated_at = :now "
            "WHERE id = :id AND status = 'queued'"
        ), {"id": row[0], "now": now})
        return int(row[0]) if updated.rowcount else None


def _job_project_id(session: Session, job: IngestionJob) -> Optional[int]:
    if job.project_id:
        return job.project_id
    if job.document_id:
        from app.models.rag import Document
        doc = session.get(Document, job.document_id)
        return doc.project_id if doc else None
    return None


def run_job(job_id: int) -> None:
    with Session(engine) as session:
        job = session.get(IngestionJob, job_id)
        if job is None:
            return
        fn = _handlers.get(job.kind)
        try:
            if fn is None:
                raise RuntimeError(f"No handler for job kind '{job.kind}'")
            # LLM calls made while indexing use the owning workspace's keys.
            with llm_keys.scope(llm_keys.scope_for_project(session, _job_project_id(session, job))):
                fn(session, job)
            job = session.get(IngestionJob, job_id)
            job.status = "done"
            job.error = None
            job.file_bytes = None  # don't keep raw uploads around once indexed
        except Exception as exc:
            logger.exception("Job %s (%s) failed", job_id, getattr(job, "kind", "?"))
            session.rollback()
            job = session.get(IngestionJob, job_id)
            job.error = str(exc)[:2000]
            job.status = "queued" if job.attempts < MAX_ATTEMPTS and _is_retryable(exc) else "failed"
            if job.status == "failed":
                job.file_bytes = None
                _on_final_failure(session, job, exc)
        job.updated_at = datetime.utcnow()
        session.add(job)
        session.commit()


def _is_retryable(exc: Exception) -> bool:
    # Parse/validation errors won't succeed on retry; transient errors might.
    return not isinstance(exc, (ValueError, KeyError))


def _on_final_failure(session: Session, job: IngestionJob, exc: Exception) -> None:
    if job.document_id is None:
        return
    from app.models.rag import Document
    doc = session.get(Document, job.document_id)
    if doc and doc.processing_status not in ("quarantined",):
        doc.processing_status = "failed"
        doc.processing_error = str(exc)[:2000]
        doc.processed = False
        session.add(doc)


def drain(max_jobs: int = 1000) -> int:
    """Process queued jobs synchronously (used by tests and CLI scripts)."""
    done = 0
    while done < max_jobs:
        job_id = _claim_next()
        if job_id is None:
            break
        run_job(job_id)
        done += 1
    return done


def recover_stale_jobs() -> None:
    cutoff = datetime.utcnow() - STALE_AFTER
    with Session(engine) as session:
        stale = session.exec(
            select(IngestionJob).where(IngestionJob.status == "running").where(IngestionJob.updated_at < cutoff)
        ).all()
        for job in stale:
            job.status = "queued" if job.attempts < MAX_ATTEMPTS else "failed"
            job.updated_at = datetime.utcnow()
            session.add(job)
        session.commit()


def _try_claim(n: int) -> Optional[int]:
    try:
        return _claim_next()
    except Exception:
        logger.exception("Worker %d could not claim a job", n)
        return None


# After a wake-up, re-check briefly: enqueue() signals before the caller's
# transaction commits, so the first look can miss a job that lands moments later.
_WAKE_RECHECK_DELAYS = (0.0, 0.2, 0.5, 1.0, 1.5)


def _worker_loop(n: int) -> None:
    logger.info("Ingestion worker %d started", n)
    while not _stop.is_set():
        job_id = _try_claim(n)
        if job_id is None:
            woke = _wake.wait(timeout=settings.worker_idle_poll_seconds)
            _wake.clear()
            if woke:
                for delay in _WAKE_RECHECK_DELAYS:
                    time.sleep(delay)
                    job_id = _try_claim(n)
                    if job_id is not None:
                        break
            if job_id is None:
                continue
        run_job(job_id)


def start_workers(count: int) -> None:
    from app.jobs import tasks  # noqa: F401  (registers handlers)
    if _threads:
        return
    recover_stale_jobs()
    _stop.clear()
    for n in range(count):
        t = threading.Thread(target=_worker_loop, args=(n,), name=f"ingest-worker-{n}", daemon=True)
        t.start()
        _threads.append(t)


def stop_workers(timeout: float = 5.0) -> None:
    _stop.set()
    _wake.set()
    deadline = time.time() + timeout
    for t in _threads:
        t.join(timeout=max(0.0, deadline - time.time()))
    _threads.clear()
