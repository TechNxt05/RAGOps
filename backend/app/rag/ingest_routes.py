from __future__ import annotations

import io
import math
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sqlmodel import Session, select

from app.auth.deps import get_current_admin, get_current_user
from app.db import engine, get_session
from app.models.rag import Chunk, Document, RAGConfig
from app.models.user import User, UserRole
from app.rag.engine import RAGEngine

router = APIRouter(prefix="/rag/ingest", tags=["rag-ingest"])


def _process_document_job(doc_id: int) -> None:
    try:
        with Session(engine) as session:
            doc = session.get(Document, doc_id)
            if not doc:
                return
            RAGEngine(session).process_document(doc)
    except Exception as exc:
        print(f"Background document processing failed for {doc_id}: {exc}")


class RechunkBody(BaseModel):
    chunk_size: int = Field(default=512, ge=128, le=8192)
    chunk_overlap: int = Field(default=50, ge=0, le=2048)


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    project_id: int = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    content = ""
    filename = file.filename or "upload.txt"
    page_count: Optional[int] = None
    raw_bytes = b""

    if filename.lower().endswith(".pdf"):
        raw_bytes = await file.read()
        reader = PdfReader(io.BytesIO(raw_bytes))
        page_count = len(reader.pages)
        for page in reader.pages:
            content += page.extract_text() + "\n"
    else:
        raw_bytes = await file.read()
        content = raw_bytes.decode("utf-8", errors="replace")

    file_size = len(raw_bytes) if raw_bytes else len(content.encode("utf-8"))

    doc = Document(
        filename=filename,
        content=content,
        processed=False,
        project_id=project_id,
        file_size_bytes=file_size,
        page_count=page_count,
        processing_status="pending",
        uploaded_by=current_user.id,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    background_tasks.add_task(_process_document_job, doc.id)

    return {
        "message": "Document queued for processing",
        "doc_id": doc.id,
        "status": doc.processing_status,
    }


@router.get("/", response_model=list[Document])
def list_documents(
    project_id: int,
    include_inactive: bool = False,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    q = select(Document).where(Document.project_id == project_id)
    if not (include_inactive and current_user.role == UserRole.ADMIN):
        q = q.where(Document.is_active == True)
    return session.exec(q.order_by(Document.uploaded_at.desc())).all()


@router.get("/documents/{doc_id}/status")
def get_document_status(
    doc_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "status": doc.processing_status,
        "error": doc.processing_error,
        "processed": doc.processed,
    }


@router.get("/documents/{doc_id}/chunks")
def get_document_chunks_page(
    doc_id: int,
    page: int = 1,
    limit: int = 20,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    all_chunks = session.exec(select(Chunk).where(Chunk.document_id == doc_id)).all()
    total_count = len(all_chunks)
    offset = max(0, (page - 1) * limit)
    chunks = all_chunks[offset : offset + limit]
    return {
        "chunks": [
            {"id": c.id, "content": c.content, "token_count": max(1, len(c.content) // 4)}
            for c in chunks
        ],
        "total": total_count,
        "page": page,
        "pages": max(1, math.ceil(total_count / limit)) if limit else 1,
    }


@router.delete("/documents/{doc_id}")
def delete_document(
    doc_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_admin),
):
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.is_active = False
    session.add(doc)
    session.commit()
    RAGEngine(session).rebuild_full_index()
    return {"deleted": True, "doc_id": doc_id}


def _rechunk_job(doc_id: int, chunk_size: int, chunk_overlap: int) -> None:
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc or not doc.content:
            return
        old_chunks = session.exec(select(Chunk).where(Chunk.document_id == doc_id)).all()
        for ch in old_chunks:
            session.delete(ch)
        session.commit()

        active = session.exec(
            select(RAGConfig)
            .where(RAGConfig.project_id == doc.project_id)
            .where(RAGConfig.is_active == True)
            .order_by(RAGConfig.created_at.desc())
        ).first()
        if active:
            active.chunk_size = chunk_size
            active.chunk_overlap = chunk_overlap
            session.add(active)
            session.commit()

        doc.processed = False
        doc.processing_status = "pending"
        doc.processing_error = None
        doc.version = (doc.version or 1) + 1
        session.add(doc)
        session.commit()

        RAGEngine(session).rebuild_full_index()
        RAGEngine(session).process_document(doc)


@router.post("/documents/{doc_id}/rechunk")
def rechunk_document(
    doc_id: int,
    background_tasks: BackgroundTasks,
    body: RechunkBody,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_admin),
):
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not doc.is_active:
        raise HTTPException(status_code=400, detail="Cannot re-chunk inactive document")

    background_tasks.add_task(_rechunk_job, doc_id, body.chunk_size, body.chunk_overlap)
    doc.processing_status = "processing"
    session.add(doc)
    session.commit()
    return {"status": "reprocessing", "doc_id": doc_id}
