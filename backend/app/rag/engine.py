import os
from typing import List
from datetime import datetime
from sqlmodel import Session, select
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.models.rag import RAGConfig, Document, Chunk
from app.db import get_session

# Global FAISS index (in-memory loaded + disk)
# On Railway, disk is ephemeral, so we ideally rebuild on startup or use a volume.
# For this implementation, we will rebuild from DB chunks on startup if valid, 
# or just rely on the 'all documents processed' state.
# But "Re-index required" implies we might destroy and recreate.

VECTOR_STORE_PATH = "faiss_index"

class RAGEngine:
    def __init__(self, session: Session):
        self.session = session
        # Use Local Embeddings to avoid Quota issues
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
    def get_active_config(self, project_id: int) -> RAGConfig:
        config = self.session.exec(select(RAGConfig).where(RAGConfig.project_id == project_id).where(RAGConfig.is_active == True)).first()
        if not config:
            # Fallback or create default? For now raise error
            raise ValueError(f"No active RAG configuration found for Project ID {project_id}")
        return config

    def process_document(self, document: Document):
        if not document.project_id:
             raise ValueError("Document must have a project_id")
             
        try:
            config = self.get_active_config(document.project_id)
        except ValueError:
            # Create default config if missing?
            config = RAGConfig(project_id=document.project_id)
            self.session.add(config)
            self.session.commit()
            self.session.refresh(config)
        
        # 1. Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
        
        texts = text_splitter.split_text(document.content)
        
        # 2. Embed and Store
        if os.path.exists(VECTOR_STORE_PATH):
            try:
                vector_store = FAISS.load_local(VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization=True)
            except:
                vector_store = None
        else:
            vector_store = None
            
        metadatas = [{"source": document.filename, "doc_id": document.id, "project_id": document.project_id} for _ in texts]
        
        if not vector_store:
            vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            vector_store.add_texts(texts, metadatas=metadatas)
            
        vector_store.save_local(VECTOR_STORE_PATH)
        
        # 3. Store Chunks in DB
        for txt in texts:
            chunk = Chunk(
                document_id=document.id,
                content=txt,
                created_at=datetime.utcnow()
            )
            self.session.add(chunk)
        
        document.processed = True
        self.session.add(document)
        self.session.commit()
    
    def reindex_all(self):
        # TODO: Implement reindex per project or global
        pass
        
    def search(self, query: str, project_id: int, k: int = 4, score_threshold: float = 0.0):
        if not os.path.exists(VECTOR_STORE_PATH):
            return []
        try:
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization=True)
            
            # Using similarity_search_with_score to avoid LangChain's strict 0-1 relevance check logic
            # which fails with some distance metrics (like L2 or unnormalized attributes).
            results_with_score = vector_store.similarity_search_with_score(
                query, 
                k=k, 
                filter={"project_id": project_id}
            )
            
            # FAISS L2 distance: Lower is better. 0 is perfect match.
            # But LangChain might normalize or not depending on the specific wrapper.
            # Usually for standard FAISS it returns distance.
            
            # Simple manual threshold if needed, but for "debug search" we definitely want to see raw results.
            # If the user passed a threshold like 0.0 (default), it might suppress everything if treating as "min similarity".
            # For debugging, we return everything found by k.
            
            # Normalize to match expected interface (Doc, score)
            return results_with_score
        except Exception as e:
            import logging
            import shutil
            logging.error(f"Error loading FAISS index: {e}")
            logging.warning("Deleting corrupt FAISS index to allow clean rebuild on next ingestion.")
            try:
                if os.path.exists(VECTOR_STORE_PATH):
                    shutil.rmtree(VECTOR_STORE_PATH)
            except Exception as e2:
                logging.error(f"Failed to delete corrupt index: {e2}")
            
            return []
        # Results is list of (Document, score)
        return results

