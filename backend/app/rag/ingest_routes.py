from __future__ import annotations

import io
import math
import os
import tempfile
import hashlib
from datetime import datetime
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
from app.services.ingestion_scanner import IngestionScanner
from app.services.docling_parser import DoclingParser

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
    raw_bytes = await file.read()
    
    # Delta Re-indexing: Check if exact same file is uploaded twice
    doc_hash = hashlib.sha256(raw_bytes).hexdigest()
    existing_doc = session.exec(
        select(Document)
        .where(Document.project_id == project_id)
        .where(Document.document_hash == doc_hash)
        .where(Document.is_active == True)
    ).first()
    
    if existing_doc:
        return {
            "message": "Document unchanged, skipping re-index",
            "doc_id": existing_doc.id,
            "status": "complete",
            "delta_index_stats": {
                "total_chunks": existing_doc.chunk_count or 0,
                "added": 0,
                "updated": 0,
                "deleted": 0,
                "unchanged": existing_doc.chunk_count or 0,
                "indexed_at": datetime.utcnow().isoformat()
            }
        }

    docling_parser = DoclingParser()
    parsing_method = "basic"
    parsed_chunks_json = None
    parsed_res = None

    if filename.lower().endswith(".pdf"):
        # Write bytes to temp file so Docling can parse it
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        
        try:
            if docling_parser.is_available():
                parsed_res = docling_parser.parse(tmp_path)
                parsing_method = parsed_res["parsing_method"]
                page_count = parsed_res["page_count"]
                
                # Reconstruct plain text content
                content = ""
                for b in parsed_res["text_blocks"]:
                    content += b["content"] + "\n"
                for t in parsed_res["tables"]:
                    content += t["content"] + "\n"
                
                parsed_chunks_json = {"chunks": docling_parser.to_chunks(parsed_res)}
            else:
                reader = PdfReader(io.BytesIO(raw_bytes))
                page_count = len(reader.pages)
                for page in reader.pages:
                    content += page.extract_text() + "\n"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        content = raw_bytes.decode("utf-8", errors="replace")

    # Ingestion PII/Secrets Scanning
    scanner = IngestionScanner()
    scan_result = scanner.scan_document(content)
    
    action_taken = "none"
    warning_msg = None
    redaction_log = None
    
    if scanner.should_quarantine(scan_result):
        # Quarantine critical secret uploads (don't chunk)
        doc = Document(
            filename=filename,
            content="",  # Blank content
            processed=False,
            project_id=project_id,
            file_size_bytes=len(raw_bytes),
            page_count=page_count,
            processing_status="quarantined",
            uploaded_by=current_user.id,
            document_hash=doc_hash,
            parsing_method=parsing_method
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)
        
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Critical secret detected. Document quarantined.",
                "doc_id": doc.id,
                "status": "quarantined",
                "ingestion_scan": {
                    "has_secrets": True,
                    "has_critical_secrets": True,
                    "total_findings": scan_result["total_findings"],
                    "findings_by_severity": scan_result["findings_by_severity"],
                    "action_taken": "quarantined"
                }
            }
        )
    
    if scan_result["has_secrets"]:
        # Redact PII
        redact_res = scanner.redact_document(content)
        content = redact_res["redacted_text"]
        redaction_log = redact_res["redaction_log"]
        action_taken = "redacted"
        warning_msg = "Document contained PII which was redacted"
        
        if parsed_chunks_json and "chunks" in parsed_chunks_json:
            for ch in parsed_chunks_json["chunks"]:
                ch_redact = scanner.redact_document(ch["content"])
                ch["content"] = ch_redact["redacted_text"]

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
        document_hash=doc_hash,
        parsing_method=parsing_method,
        redaction_log=redaction_log,
        parsed_chunks_json=parsed_chunks_json
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    background_tasks.add_task(_process_document_job, doc.id)

    estimated_chunks = parsed_chunks_json["chunks"] if parsed_chunks_json else []

    return {
        "message": warning_msg or "Document queued for processing",
        "doc_id": doc.id,
        "status": doc.processing_status,
        "ingestion_scan": {
            "has_secrets": scan_result["has_secrets"],
            "has_critical_secrets": False,
            "total_findings": scan_result["total_findings"],
            "findings_by_severity": scan_result["findings_by_severity"],
            "action_taken": action_taken,
            "redaction_count": len(redaction_log) if redaction_log else 0
        },
        "parsing_stats": {
            "parsing_method": parsing_method,
            "page_count": page_count or 0,
            "table_count": len(parsed_res["tables"]) if parsed_res and "tables" in parsed_res else 0,
            "heading_count": len(parsed_res["headings"]) if parsed_res and "headings" in parsed_res else 0,
            "chunk_count": len(estimated_chunks) if estimated_chunks else 0
        }
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
