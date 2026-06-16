import re
from typing import List, Optional, Any, Set
from sqlmodel import Session, select
from app.models.rag import Chunk, Document

class AbsenceProver:
    def __init__(self, session: Session):
        self.session = session
    
    def build_keyword_set(self, query: str) -> List[str]:
        """
        Extract content keywords from query for full-corpus scan.
        Remove stopwords, keep meaningful terms.
        """
        stopwords = {
            "what", "is", "the", "a", "an", "in", "of", "for", "and",
            "or", "to", "how", "when", "where", "who", "which", "are",
            "does", "do", "can", "will", "has", "have", "been", "be",
            "was", "were", "that", "this", "with", "by", "from", "at"
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        keywords = [w for w in words if w not in stopwords]
        
        return keywords
    
    def scan_corpus(
        self,
        keywords: List[str],
        project_id: int,
        min_keyword_matches: int = 1
    ) -> dict:
        """
        Run literal keyword scan across ALL chunks in project corpus.
        This is the deterministic absence proof.
        """
        # Query ALL chunks and documents for this project
        statement = (
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.project_id == project_id)
            .where(Document.is_active == True)
        )
        
        results = self.session.exec(statement).all()
        total = len(results)
        
        matching_chunks = []
        matching_sources = set()
        
        for chunk, doc in results:
            content_lower = chunk.content.lower()
            matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            if matches >= min_keyword_matches:
                matching_chunks.append((chunk, doc))
                if doc.filename:
                    matching_sources.add(doc.filename)
        
        absence_proven = len(matching_chunks) == 0
        
        return {
            "keywords_searched": keywords,
            "total_chunks_scanned": total,
            "matching_chunk_count": len(matching_chunks),
            "absence_proven": absence_proven,
            "matching_sources": list(matching_sources),
            "matching_chunks_sample": [
                {
                    "chunk_id": chunk.doc_id_version or f"chunk_{chunk.id}",
                    "source": doc.filename or "Unknown",
                    "preview": chunk.content[:150]
                }
                for chunk, doc in matching_chunks[:5]  # Show max 5 samples
            ]
        }
    
    def prove_or_retry(
        self,
        query: str,
        project_id: int,
        hybrid_search_fn,
        top_k: int = 5
    ) -> dict:
        """
        Called when confidence gate returns low confidence or LLM says not found.
        """
        keywords = self.build_keyword_set(query)
        
        if not keywords:
            return {
                "action": "inconclusive",
                "absence_proof": None,
                "retry_chunks": None
            }
        
        proof = self.scan_corpus(keywords, project_id)
        
        if proof["absence_proven"]:
            return {
                "action": "proven_absent",
                "absence_proof": proof,
                "retry_chunks": None
            }
        
        # Keywords found somewhere in corpus but not retrieved
        # Trigger targeted retry
        try:
            retry_chunks = hybrid_search_fn(
                query=query, 
                project_id=project_id, 
                k=top_k,
                filter_sources=set(proof["matching_sources"])
            )
        except Exception as e:
            print(f"Error executing targeted retry: {e}")
            retry_chunks = []
            
        return {
            "action": "retry_triggered" if retry_chunks else "proven_absent",
            "absence_proof": proof,
            "retry_chunks": retry_chunks if retry_chunks else None
        }
