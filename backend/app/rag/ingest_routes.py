from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlmodel import Session, select
from app.db import get_session
from app.models.rag import Document
from app.models.user import User
from app.auth.deps import get_current_admin, get_current_user
from app.rag.engine import RAGEngine
import shutil
import os
from pypdf import PdfReader

router = APIRouter(prefix="/rag/ingest", tags=["rag-ingest"])

def process_file_background(doc_id: int, session_generator):
    # We need a new session for background task
    session = next(session_generator())
    try:
        doc = session.get(Document, doc_id)
        if doc and not doc.processed:
            engine = RAGEngine(session)
            engine.process_document(doc)
    finally:
        session.close()

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    project_id: int = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    # 1. Read file content
    content = ""
    filename = file.filename
    
    if filename.endswith(".pdf"):
        reader = PdfReader(file.file)
        for page in reader.pages:
            content += page.extract_text() + "\n"
    else:
        content = (await file.read()).decode("utf-8")
        
    # 2. Save Document to DB
    doc = Document(filename=filename, content=content, processed=False, project_id=project_id)
    session.add(doc)
    session.commit()
    session.refresh(doc)
    
    # 3. Trigger Processing
    engine = RAGEngine(session)
    engine.process_document(doc)
    
    return {"message": "Document uploaded and processed successfully", "doc_id": doc.id}

@router.get("/", response_model=list[Document])
def list_documents(project_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    # Check if project exists (optional but good)
    docs = session.exec(select(Document).where(Document.project_id == project_id).order_by(Document.uploaded_at.desc())).all()
    return docs
