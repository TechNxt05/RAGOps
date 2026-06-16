import hashlib
from typing import List, Dict, Optional
from datetime import datetime
from sqlmodel import Session, select
from app.models.rag import Chunk, Document, Project

VECTOR_STORE_PATH = "faiss_index"

class DeltaIndexer:
    def __init__(self, session: Session, vector_store, embedding_model):
        self.session = session
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def compute_chunk_hash(self, chunk_text: str) -> str:
        """SHA-256 hash of chunk content."""
        return hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

    def compute_document_hash(self, document_content: bytes) -> str:
        """SHA-256 hash of full document bytes."""
        return hashlib.sha256(document_content).hexdigest()

    def get_existing_hashes(self, document_id: int) -> Dict[int, str]:
        """
        Returns dict of {chunk_index: content_hash} for all existing
        chunks of this document.
        """
        existing = {}
        statement = select(Chunk).where(Chunk.document_id == document_id)
        chunks = self.session.exec(statement).all()
        for chunk in chunks:
            if chunk.chunk_index is not None:
                existing[chunk.chunk_index] = chunk.content_hash or ""
        return existing

    def _get_doc_project_id(self, document_id: int) -> Optional[int]:
        doc = self.session.get(Document, document_id)
        return doc.project_id if doc else None

    def _get_doc_filename(self, document_id: int) -> str:
        doc = self.session.get(Document, document_id)
        return doc.filename if doc else "unknown"

    def delta_index(
        self, 
        document_id: int,
        new_chunks: List[dict],
        force_full: bool = False
    ) -> dict:
        """
        Compare new chunks against existing indexed chunks.
        Only re-embed chunks that have changed.
        Delete chunks that no longer exist.
        
        new_chunks: List of {"index": int, "text": str, "metadata": dict}
        """
        existing_hashes = {} if force_full else self.get_existing_hashes(document_id)
        
        new_chunk_map = {}
        for chunk in new_chunks:
            chunk_hash = self.compute_chunk_hash(chunk["text"])
            new_chunk_map[chunk["index"]] = {
                **chunk,
                "content_hash": chunk_hash,
                "doc_id_version": f"{document_id}:{chunk['index']}:{chunk_hash[:8]}"
            }
        
        # Determine what changed
        existing_indices = set(existing_hashes.keys())
        new_indices = set(new_chunk_map.keys())
        
        to_add = []      # new chunks not in existing
        to_update = []   # chunks that exist but hash changed
        to_delete = []   # chunks in existing but not in new
        unchanged = []   # chunks with same hash
        
        for idx in new_indices:
            if idx not in existing_indices:
                to_add.append(new_chunk_map[idx])
            elif existing_hashes[idx] != new_chunk_map[idx]["content_hash"]:
                to_update.append(new_chunk_map[idx])
            else:
                unchanged.append(idx)
        
        for idx in existing_indices:
            if idx not in new_indices:
                to_delete.append(idx)
        
        # Execute changes in transactions
        # 1. Delete removed chunks from vector store + DB
        if to_delete:
            self._delete_chunks(document_id, to_delete)
        
        # 2. Update changed chunks (delete old + add new)
        if to_update:
            self._delete_chunks(document_id, [c["index"] for c in to_update])
            self._embed_and_store(document_id, to_update)
        
        # 3. Add new chunks
        if to_add:
            self._embed_and_store(document_id, to_add)
        
        # Update kb_version on project
        project_id = self._get_doc_project_id(document_id)
        if project_id:
            project = self.session.get(Project, project_id)
            if project:
                project.kb_version = (project.kb_version or 1) + 1
                project.kb_version_updated_at = datetime.utcnow()
                self.session.add(project)
        
        self.session.commit()
        
        return {
            "total_chunks": len(new_chunks),
            "added": len(to_add),
            "updated": len(to_update),
            "deleted": len(to_delete),
            "unchanged": len(unchanged),
            "indexed_at": datetime.utcnow().isoformat()
        }

    def _embed_and_store(self, document_id: int, chunks: List[dict]) -> None:
        """Embed chunks and store in FAISS + PostgreSQL."""
        if not chunks:
            return
            
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.embed_documents(texts)
        
        project_id = self._get_doc_project_id(document_id)
        filename = self._get_doc_filename(document_id)
        
        for chunk, embedding in zip(chunks, embeddings):
            # Store in FAISS vector store
            if self.vector_store:
                self.vector_store.add_texts(
                    texts=[chunk["text"]],
                    metadatas=[{
                        "document_id": document_id,
                        "doc_id": document_id,
                        "project_id": project_id,
                        "source": filename,
                        "content_hash": chunk["content_hash"],
                        "doc_id_version": chunk["doc_id_version"],
                        **chunk.get("metadata", {})
                    }],
                    ids=[chunk["doc_id_version"]]
                )
                self.vector_store.save_local(VECTOR_STORE_PATH)
            
            # Upsert chunk record in PostgreSQL
            db_chunk = self.session.exec(
                select(Chunk)
                .where(Chunk.document_id == document_id)
                .where(Chunk.chunk_index == chunk["index"])
            ).first()
            
            if not db_chunk:
                db_chunk = Chunk(
                    document_id=document_id,
                    content=chunk["text"],
                    chunk_index=chunk["index"],
                    content_hash=chunk["content_hash"],
                    doc_id_version=chunk["doc_id_version"],
                    chunk_version=1
                )
            else:
                db_chunk.content = chunk["text"]
                db_chunk.content_hash = chunk["content_hash"]
                db_chunk.doc_id_version = chunk["doc_id_version"]
                db_chunk.chunk_version += 1
            
            self.session.add(db_chunk)

    def _delete_chunks(self, document_id: int, chunk_indices: List[int]) -> None:
        """Delete specific chunk indices from FAISS + PostgreSQL."""
        statement = select(Chunk).where(Chunk.document_id == document_id).where(Chunk.chunk_index.in_(chunk_indices))
        chunks_to_delete = self.session.exec(statement).all()
        
        ids_to_delete = [c.doc_id_version for c in chunks_to_delete if c.doc_id_version]
        
        if ids_to_delete and self.vector_store:
            try:
                self.vector_store.delete(ids=ids_to_delete)
                self.vector_store.save_local(VECTOR_STORE_PATH)
            except Exception as e:
                print(f"Error deleting from FAISS: {e}")
                
        for chunk in chunks_to_delete:
            self.session.delete(chunk)
