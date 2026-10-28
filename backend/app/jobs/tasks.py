"""Job handlers: parse -> scan -> chunk -> embed -> index."""

import logging

from sqlmodel import Session, select

from app.jobs.queue import handler
from app.models.rag import Chunk, Document, IngestionJob, RAGConfig
from app.rag import vector_store
from app.rag.engine import RAGEngine, bump_kb_version

logger = logging.getLogger(__name__)


@handler("ingest_file")
def ingest_file(session: Session, job: IngestionJob) -> None:
    from app.services.document_parser import parse_file
    from app.services.ingestion_scanner import IngestionScanner

    doc = session.get(Document, job.document_id)
    if doc is None or not doc.is_active:
        return
    if job.file_bytes is None:
        raise ValueError("Upload payload missing")

    doc.processing_status = "parsing"
    session.add(doc)
    session.commit()

    try:
        parsed = parse_file(doc.filename, job.file_bytes)
    except Exception as exc:
        doc.processing_status = "failed"
        doc.processing_error = f"Could not read file: {exc}"[:2000]
        session.add(doc)
        session.commit()
        raise ValueError(doc.processing_error) from exc

    content = parsed["content"]
    parsed_chunks_json = parsed.get("parsed_chunks_json")

    scanner = IngestionScanner()
    scan = scanner.scan_document(content)
    doc.ingestion_scan = {
        "has_secrets": scan["has_secrets"],
        "total_findings": scan["total_findings"],
        "findings_by_severity": scan["findings_by_severity"],
    }
    doc.page_count = parsed.get("page_count")
    doc.parsing_method = parsed.get("parsing_method")

    if scanner.should_quarantine(scan):
        # Critical secrets (API keys, private keys...): never chunk or embed.
        doc.content = ""
        doc.processing_status = "quarantined"
        doc.processing_error = "Critical secret detected. Document quarantined and not indexed."
        doc.ingestion_scan["action_taken"] = "quarantined"
        session.add(doc)
        session.commit()
        return

    if scan["has_secrets"]:
        redacted = scanner.redact_document(content)
        content = redacted["redacted_text"]
        doc.redaction_log = redacted["redaction_log"]
        doc.ingestion_scan["action_taken"] = "redacted"
        doc.ingestion_scan["redaction_count"] = len(redacted["redaction_log"] or [])
        if parsed_chunks_json and "chunks" in parsed_chunks_json:
            for ch in parsed_chunks_json["chunks"]:
                ch["content"] = scanner.redact_document(ch["content"])["redacted_text"]
    else:
        doc.ingestion_scan["action_taken"] = "none"

    doc.content = content
    doc.parsed_chunks_json = parsed_chunks_json
    session.add(doc)
    session.commit()

    RAGEngine(session).process_document(doc)


@handler("reindex_document")
def reindex_document(session: Session, job: IngestionJob) -> None:
    """Re-chunk one document (optionally with new chunk settings) and re-index it."""
    doc = session.get(Document, job.document_id)
    if doc is None or not doc.is_active or not (doc.content or doc.parsed_chunks_json):
        return
    payload = job.payload or {}
    if "chunk_size" in payload or "chunk_overlap" in payload:
        config = session.exec(
            select(RAGConfig)
            .where(RAGConfig.project_id == doc.project_id)
            .where(RAGConfig.is_active == True)  # noqa: E712
            .order_by(RAGConfig.created_at.desc())
        ).first()
        if config is None:
            config = RAGConfig(project_id=doc.project_id)
        config.chunk_size = int(payload.get("chunk_size", config.chunk_size))
        config.chunk_overlap = int(payload.get("chunk_overlap", config.chunk_overlap))
        session.add(config)
        # Re-chunking with new settings must use the adaptive chunker, not the
        # Docling pre-chunked layout.
        if payload.get("force_rechunk"):
            doc.parsing_method = doc.parsing_method if doc.parsing_method != "docling" else "docling-text"
        doc.version = (doc.version or 1) + 1
        session.add(doc)
        session.commit()
    RAGEngine(session).process_document(doc)


@handler("reindex_project")
def reindex_project(session: Session, job: IngestionJob) -> None:
    RAGEngine(session).rebuild_full_index(job.project_id)


def deactivate_document(session: Session, doc: Document) -> None:
    """Soft-delete a document: hide it and drop its chunks/vectors immediately.
    No re-embedding of anything else is needed (the old FAISS design rebuilt the
    whole index on every delete)."""
    doc.is_active = False
    doc.processing_status = "deleted"
    session.add(doc)
    chunk_ids = session.exec(select(Chunk.id).where(Chunk.document_id == doc.id)).all()
    vector_store.delete_for_chunks(session, list(chunk_ids))
    for chunk in session.exec(select(Chunk).where(Chunk.document_id == doc.id)).all():
        session.delete(chunk)
    if doc.project_id:
        bump_kb_version(session, doc.project_id)
