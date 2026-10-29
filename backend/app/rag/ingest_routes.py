from __future__ import annotations

import hashlib
import math
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlmodel import Session, func, select

from app.auth.access import project_workspace, require_project
from app.auth.deps import get_current_user
from app.services import workspaces
from app.core.config import settings
from app.core.rate_limit import rate_limiter
from app.db import get_session
from app.jobs import queue
from app.jobs.tasks import deactivate_document
from app.models.rag import Chunk, Document, IngestionJob
from app.models.user import User, UserRole
from app.services.document_parser import check_upload

router = APIRouter(prefix="/rag/ingest", tags=["rag-ingest"])

# Statuses meaning "this exact file is already (being) indexed" for de-duplication.
_LIVE_STATUSES = ("pending", "queued", "parsing", "processing", "complete")
_UPLOAD_CHUNK = 1024 * 1024


class RechunkBody(BaseModel):
    chunk_size: int = Field(default=512, ge=128, le=8192)
    chunk_overlap: int = Field(default=50, ge=0, le=2048)


class BulkIds(BaseModel):
    doc_ids: List[int] = Field(min_length=1, max_length=500)


async def _read_limited(upload: UploadFile, max_bytes: int) -> Optional[bytes]:
    """Read an upload but stop as soon as it exceeds the limit."""
    buf = bytearray()
    while True:
        chunk = await upload.read(_UPLOAD_CHUNK)
        if not chunk:
            return bytes(buf)
        buf.extend(chunk)
        if len(buf) > max_bytes:
            return None


@router.post("/upload")
async def upload_documents(
    project_id: int = Form(...),
    files: List[UploadFile] = File(default=[]),
    file: Optional[UploadFile] = File(default=None),  # legacy single-file field
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Accept one or many files. Each file is validated, de-duplicated and queued
    for background parsing + indexing; the response returns immediately with a
    per-file result. A bad file never fails the whole batch."""
    project = require_project(session, current_user, project_id, write=True)
    rate_limiter.check(f"upload:{current_user.id}", settings.rate_limit_upload_per_min)

    uploads = list(files) + ([file] if file is not None else [])
    if not uploads:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(uploads) > settings.max_files_per_upload:
        raise HTTPException(status_code=400, detail=f"At most {settings.max_files_per_upload} files per upload")
    workspaces.check_document_limit(session, project_workspace(session, project), len(uploads))

    max_bytes = settings.max_upload_mb * 1024 * 1024
    results = []
    for upload in uploads:
        filename = (upload.filename or "upload.txt").replace("\\", "/").split("/")[-1][:255]
        raw = await _read_limited(upload, max_bytes)
        error = "File exceeds the size limit" if raw is None else check_upload(filename, len(raw), max_bytes)
        if error:
            results.append({"filename": filename, "status": "rejected", "error": error})
            continue

        doc_hash = hashlib.sha256(raw).hexdigest()
        existing = session.exec(
            select(Document)
            .where(Document.project_id == project_id)
            .where(Document.document_hash == doc_hash)
            .where(Document.is_active == True)  # noqa: E712
        ).all()
        live = next((d for d in existing if d.processing_status in _LIVE_STATUSES), None)
        if live:
            results.append({
                "filename": filename, "doc_id": live.id, "status": "duplicate",
                "message": f"Identical file already in project as '{live.filename}'",
            })
            continue
        # A previous failed/quarantined copy of this file is superseded by the retry.
        for old in existing:
            old.is_active = False
            session.add(old)

        doc = Document(
            filename=filename,
            content="",
            processed=False,
            project_id=project_id,
            file_size_bytes=len(raw),
            processing_status="queued",
            uploaded_by=current_user.id,
            document_hash=doc_hash,
        )
        session.add(doc)
        session.flush()
        queue.enqueue(session, "ingest_file", project_id=project_id, document_id=doc.id,
                      payload={"filename": filename}, file_bytes=raw, commit=False)
        session.commit()
        results.append({"filename": filename, "doc_id": doc.id, "status": "queued"})

    queued = sum(1 for r in results if r["status"] == "queued")
    first = next((r for r in results if r.get("doc_id")), {})
    return {
        "message": f"{queued} of {len(results)} file(s) queued for processing",
        "results": results,
        # Legacy single-file response fields
        "doc_id": first.get("doc_id"),
        "status": first.get("status"),
    }


@router.get("/", response_model=list[Document])
def list_documents(
    project_id: int,
    include_inactive: bool = False,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    require_project(session, current_user, project_id)
    q = select(Document).where(Document.project_id == project_id)
    if not (include_inactive and current_user.role == UserRole.ADMIN):
        q = q.where(Document.is_active == True)  # noqa: E712
    docs = session.exec(q.order_by(Document.uploaded_at.desc())).all()
    # Full text and parse artefacts can be megabytes; the list view doesn't need them.
    return [d.model_copy(update={"content": None, "parsed_chunks_json": None}) for d in docs]


@router.get("/status")
def get_documents_status(
    project_id: int,
    doc_ids: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Batched status poll: ?doc_ids=1,2,3 (one request for the whole upload queue)."""
    require_project(session, current_user, project_id)
    ids = [int(x) for x in doc_ids.split(",") if x.strip().isdigit()][:500]
    if not ids:
        return {"documents": []}
    docs = session.exec(
        select(Document).where(Document.id.in_(ids)).where(Document.project_id == project_id)
    ).all()
    return {"documents": [
        {
            "doc_id": d.id, "filename": d.filename, "status": d.processing_status,
            "error": d.processing_error, "processed": d.processed, "chunk_count": d.chunk_count,
            "ingestion_scan": d.ingestion_scan,
        }
        for d in docs
    ]}


def _doc_for_user(session: Session, user: User, doc_id: int, write: bool = False) -> Document:
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    require_project(session, user, doc.project_id, write=write)
    return doc


@router.get("/documents/{doc_id}/status")
def get_document_status(
    doc_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    doc = _doc_for_user(session, current_user, doc_id)
    return {
        "status": doc.processing_status,
        "error": doc.processing_error,
        "processed": doc.processed,
        "ingestion_scan": doc.ingestion_scan,
    }


@router.get("/documents/{doc_id}/chunks")
def get_document_chunks_page(
    doc_id: int,
    page: int = 1,
    limit: int = 20,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    _doc_for_user(session, current_user, doc_id, write=True)
    limit = max(1, min(limit, 200))
    page = max(1, page)
    total_count = session.exec(select(func.count(Chunk.id)).where(Chunk.document_id == doc_id)).one()
    chunks = session.exec(
        select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.chunk_index, Chunk.id)
        .offset((page - 1) * limit).limit(limit)
    ).all()
    return {
        "chunks": [
            {"id": c.id, "content": c.content, "token_count": max(1, len(c.content) // 4)}
            for c in chunks
        ],
        "total": total_count,
        "page": page,
        "pages": max(1, math.ceil(total_count / limit)),
    }


@router.delete("/documents/{doc_id}")
def delete_document(
    doc_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    doc = _doc_for_user(session, current_user, doc_id, write=True)
    deactivate_document(session, doc)
    session.commit()
    return {"deleted": True, "doc_id": doc_id}


@router.post("/documents/bulk-delete")
def bulk_delete_documents(
    body: BulkIds,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    deleted, failed = [], []
    for doc_id in body.doc_ids:
        try:
            doc = _doc_for_user(session, current_user, doc_id, write=True)
            deactivate_document(session, doc)
            deleted.append(doc_id)
        except HTTPException as exc:
            failed.append({"doc_id": doc_id, "error": exc.detail})
    session.commit()
    return {"deleted": deleted, "failed": failed}


@router.post("/documents/{doc_id}/rechunk")
def rechunk_document(
    doc_id: int,
    body: RechunkBody,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    doc = _doc_for_user(session, current_user, doc_id, write=True)
    if not doc.is_active:
        raise HTTPException(status_code=400, detail="Cannot re-chunk inactive document")
    if body.chunk_overlap >= body.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size")
    doc.processing_status = "queued"
    session.add(doc)
    queue.enqueue(session, "reindex_document", project_id=doc.project_id, document_id=doc.id,
                  payload={"chunk_size": body.chunk_size, "chunk_overlap": body.chunk_overlap, "force_rechunk": True},
                  commit=False)
    session.commit()
    return {"status": "reprocessing", "doc_id": doc_id}


@router.post("/projects/{project_id}/reindex")
def reindex_project(
    project_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Re-embed every chunk of a project (e.g. after changing its embedding model)."""
    require_project(session, current_user, project_id, write=True)
    job = queue.enqueue(session, "reindex_project", project_id=project_id)
    return {"status": "queued", "job_id": job.id}


@router.get("/jobs")
def list_jobs(
    project_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    require_project(session, current_user, project_id, write=True)
    jobs = session.exec(
        select(IngestionJob).where(IngestionJob.project_id == project_id).order_by(IngestionJob.id.desc()).limit(50)
    ).all()
    return [
        {"id": j.id, "kind": j.kind, "status": j.status, "document_id": j.document_id,
         "attempts": j.attempts, "error": j.error, "created_at": j.created_at, "updated_at": j.updated_at}
        for j in jobs
    ]
