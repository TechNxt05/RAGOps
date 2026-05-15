import os
import shutil
import logging
from typing import List, Tuple, Optional
from datetime import datetime
from sqlmodel import Session, select
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument

from app.models.rag import RAGConfig, Document, Chunk

VECTOR_STORE_PATH = "faiss_index"


class RAGEngine:
    def __init__(self, session: Session):
        self.session = session
        # Default fallback embeddings if no project config is found
        self._default_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

    def _get_embeddings(self, config: Optional[RAGConfig] = None):
        """Load embeddings based on config. Defaults to Google embedding-001."""
        if not config or not config.embedding_model:
            return self._default_embeddings

        if "huggingface" in config.embedding_model.lower() or "minilm" in config.embedding_model.lower():
            # Use local HuggingFace embeddings (no API cost)
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Default to Google
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

    def get_active_config(self, project_id: int) -> RAGConfig:
        config = self.session.exec(
            select(RAGConfig)
            .where(RAGConfig.project_id == project_id)
            .where(RAGConfig.is_active == True)
        ).first()
        if not config:
            raise ValueError(f"No active RAG configuration found for Project ID {project_id}")
        return config

    def _inactive_doc_ids(self, project_id: int) -> set[int]:
        rows = self.session.exec(
            select(Document.id).where(Document.project_id == project_id).where(Document.is_active == False)
        ).all()
        return {int(r) for r in rows if r is not None}

    def rebuild_full_index(self, project_id: Optional[int] = None) -> None:
        """Rebuild FAISS from all chunks belonging to active, processed documents."""
        # For simplicity in this enterprise version, we rebuild for the specific project's embeddings
        # If project_id is None, we use default embeddings
        config = None
        if project_id:
            try:
                config = self.get_active_config(project_id)
            except Exception:
                pass
        
        embeddings = self._get_embeddings(config)

        rows_query = (
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.is_active == True)
            .where(Document.processed == True)
        )
        
        if project_id:
            rows_query = rows_query.where(Document.project_id == project_id)
            
        rows = self.session.exec(rows_query).all()

        texts: List[str] = []
        metadatas: List[dict] = []
        for chunk, doc in rows:
            texts.append(chunk.content)
            metadatas.append(
                {
                    "source": doc.filename,
                    "doc_id": doc.id,
                    "project_id": doc.project_id,
                }
            )

        if not texts:
            # If no texts left for this project/global, and we are global, clear it.
            # Otherwise we'd need per-project index files. 
            # For this MVP+ we'll clear global if texts is empty.
            if os.path.exists(VECTOR_STORE_PATH):
                shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)
            return

        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vector_store.save_local(VECTOR_STORE_PATH)

    def process_document(self, document: Document) -> None:
        if not document.project_id:
            raise ValueError("Document must have a project_id")

        document.processing_status = "processing"
        document.processing_error = None
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)

        try:
            try:
                config = self.get_active_config(document.project_id)
            except ValueError:
                config = RAGConfig(project_id=document.project_id)
                self.session.add(config)
                self.session.commit()
                self.session.refresh(config)

            embeddings = self._get_embeddings(config)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                length_function=len,
            )

            content = document.content or ""
            texts = text_splitter.split_text(content)

            # Check if existing index matches current embedding model
            # For simplicity, we assume one project = one index for now
            # In a real enterprise app, we'd use project-specific index paths
            if os.path.exists(VECTOR_STORE_PATH):
                try:
                    vector_store = FAISS.load_local(
                        VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
                    )
                except Exception:
                    # If loading fails (likely embedding mismatch), we start fresh or should rebuild
                    vector_store = None
            else:
                vector_store = None

            metadatas = [
                {"source": document.filename, "doc_id": document.id, "project_id": document.project_id}
                for _ in texts
            ]

            if not vector_store:
                vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            else:
                vector_store.add_texts(texts, metadatas=metadatas)

            vector_store.save_local(VECTOR_STORE_PATH)

            for txt in texts:
                chunk = Chunk(
                    document_id=document.id,
                    content=txt,
                    created_at=datetime.utcnow(),
                )
                self.session.add(chunk)

            document.processed = True
            document.processing_status = "complete"
            document.chunk_count = len(texts)
            document.chunk_size_used = config.chunk_size
            document.embedding_model_used = config.embedding_model or "google-embedding-001"
            self.session.add(document)
            self.session.commit()
        except Exception as exc:
            document.processed = False
            document.processing_status = "failed"
            document.processing_error = str(exc)
            self.session.add(document)
            self.session.commit()
            raise

    def reindex_all(self) -> None:
        self.rebuild_full_index()

    def search(
        self, query: str, project_id: int, k: int = 4, score_threshold: float = 0.0
    ) -> List[Tuple[LCDocument, float]]:
        if not os.path.exists(VECTOR_STORE_PATH):
            return []
            
        try:
            config = self.get_active_config(project_id)
            embeddings = self._get_embeddings(config)
            
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
            )

            results_with_score = vector_store.similarity_search_with_score(
                query,
                k=k,
                filter={"project_id": project_id},
            )

            inactive = self._inactive_doc_ids(project_id)
            filtered: List[Tuple[LCDocument, float]] = []
            for doc, score in results_with_score:
                meta = doc.metadata or {}
                did = meta.get("doc_id")
                if did is not None and int(did) in inactive:
                    continue
                filtered.append((doc, score))

            return filtered
        except Exception as e:
            logging.error(f"Error loading FAISS index: {e}")
            return []
