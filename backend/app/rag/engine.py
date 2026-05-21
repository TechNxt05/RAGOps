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

from app.models.rag import RAGConfig, Document, Chunk, Project
from app.services.bm25_service import bm25_manager
from app.services.rrf_service import hybrid_search_merge
from app.services.context_pruner import context_pruner
from app.services.reranker_service import reranker_service

VECTOR_STORE_PATH = "faiss_index"


class SearchResultList(list):
    """
    Subclass of list to hold context pruning and hybrid retrieval metadata,
    allowing existing routing layers to access these statistics without breaking
    signature compatibility.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunks_before_pruning = 0
        self.chunks_after_pruning = 0
        self.pruning_reduction_pct = 0.0
        self.used_hybrid_search = False


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

    def _rebuild_bm25_for_project(self, project_id: int) -> None:
        """Fetches all active chunks for a project and builds the BM25 index."""
        try:
            rows = self.session.exec(
                select(Chunk)
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.project_id == project_id)
                .where(Document.is_active == True)
                .where(Document.processed == True)
            ).all()
            chunks = [chunk.content for chunk in rows]
            if chunks:
                bm25_manager.build_index(str(project_id), chunks)
            else:
                bm25_manager.delete_index(str(project_id))
        except Exception as e:
            logging.error(f"Error rebuilding BM25 index for project {project_id}: {e}")

    def rebuild_full_index(self, project_id: Optional[int] = None) -> None:
        """Rebuild FAISS from all chunks belonging to active, processed documents."""
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
            if os.path.exists(VECTOR_STORE_PATH):
                shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)
            if project_id:
                bm25_manager.delete_index(str(project_id))
            return

        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vector_store.save_local(VECTOR_STORE_PATH)

        # Build BM25 index for projects
        if project_id:
            self._rebuild_bm25_for_project(project_id)
        else:
            projects = self.session.exec(select(Project)).all()
            for proj in projects:
                if proj.id:
                    self._rebuild_bm25_for_project(proj.id)

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

            if os.path.exists(VECTOR_STORE_PATH):
                try:
                    vector_store = FAISS.load_local(
                        VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
                    )
                except Exception:
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

            # Build BM25 index for the project after processing
            self._rebuild_bm25_for_project(document.project_id)
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
            return SearchResultList()
            
        try:
            config = self.get_active_config(project_id)
            embeddings = self._get_embeddings(config)
            
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
            )

            # Retrieve candidate pool for hybrid, pruning, and reranking
            candidate_k = max(k * 5, 20)

            results_with_score = vector_store.similarity_search_with_score(
                query,
                k=candidate_k,
                filter={"project_id": project_id},
            )

            inactive = self._inactive_doc_ids(project_id)
            semantic_results: List[Tuple[LCDocument, float]] = []
            for doc, score in results_with_score:
                meta = doc.metadata or {}
                did = meta.get("doc_id")
                if did is not None and int(did) in inactive:
                    continue
                semantic_results.append((doc, score))

            merged_results = []
            used_hybrid = False

            # If hybrid search is enabled and the index exists, do BM25 + merge
            if config.use_hybrid_search and bm25_manager.index_exists(str(project_id)):
                bm25_res = bm25_manager.search(str(project_id), query, top_k=candidate_k)
                if bm25_res:
                    merged_results = hybrid_search_merge(
                        semantic_results=semantic_results,
                        bm25_results=bm25_res,
                        semantic_weight=config.semantic_weight,
                        bm25_weight=1.0 - config.semantic_weight
                    )
                    used_hybrid = True

            if not used_hybrid:
                merged_results = semantic_results

            # Context Pruning (using TF-IDF similarity threshold, default at least 0.1)
            prune_threshold = max(config.similarity_threshold or 0.0, 0.1)
            pruned_results, orig_count, pruned_count, reduction_pct = context_pruner.prune(
                query=query,
                chunks=merged_results,
                threshold=prune_threshold
            )

            # BGE Reranker (re-rank pruned candidates down to target k)
            final_reranked = reranker_service.rerank(
                query=query,
                chunks=pruned_results,
                top_k=k
            )

            # Wrap in SearchResultList subclass to carry metadata back to route handlers
            final_results = SearchResultList(final_reranked)
            final_results.chunks_before_pruning = orig_count
            final_results.chunks_after_pruning = pruned_count
            final_results.pruning_reduction_pct = reduction_pct
            final_results.used_hybrid_search = used_hybrid

            return final_results
        except Exception as e:
            logging.error(f"Error loading FAISS index: {e}")
            return SearchResultList()
