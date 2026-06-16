import re
from typing import List, Optional, Any

class CrossReferenceResolver:
    # Patterns that signal a cross-reference in retrieved text
    REFERENCE_PATTERNS = [
        r"(?:see|refer to|as defined in|as stated in|per|according to)\s+"
        r"(?:section|sec\.|appendix|clause|article|paragraph|part|chapter)\s*"
        r"[\d\.]+[a-z]?",
        r"(?:section|sec\.|appendix|clause|article|paragraph|part|chapter)\s*"
        r"[\d\.]+[a-z]?\s+(?:above|below|of this|herein)",
        r"as (?:described|outlined|specified|mentioned|noted) (?:above|below|herein)",
        r"see (?:above|below|attached|appendix)",
        r"\(see [\w\s\.]+\)",
    ]
    
    def detect_references(self, chunks: List[Any]) -> List[dict]:
        """
        Scan retrieved chunks for cross-references.
        Returns list of detected references with their source chunk.
        """
        references = []
        
        for chunk_idx, chunk in enumerate(chunks):
            # Parse chunk content, document_id, and doc_id_version
            content = ""
            doc_id = None
            doc_id_version = ""
            
            if isinstance(chunk, tuple):
                doc = chunk[0]
                content = doc.page_content
                doc_id = doc.metadata.get("doc_id")
                doc_id_version = doc.metadata.get("doc_id_version")
            elif hasattr(chunk, "page_content"):
                content = chunk.page_content
                doc_id = chunk.metadata.get("doc_id")
                doc_id_version = chunk.metadata.get("doc_id_version")
            elif isinstance(chunk, dict):
                content = chunk.get("content") or chunk.get("text") or ""
                doc_id = chunk.get("document_id") or chunk.get("doc_id")
                doc_id_version = chunk.get("doc_id_version") or chunk.get("id")
                
            if not content:
                continue
                
            for pattern in self.REFERENCE_PATTERNS:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    references.append({
                        "source_chunk_idx": chunk_idx,
                        "source_chunk_id": doc_id_version or f"chunk_{chunk_idx}",
                        "reference_text": match.group(0).strip(),
                        "reference_start": match.start(),
                        "reference_end": match.end(),
                        "document_id": doc_id,
                        "resolved": False,
                        "resolved_chunks": []
                    })
        
        return references
    
    def resolve_reference(
        self,
        reference: dict,
        hybrid_search_fn,
        project_id: int,
        top_k: int = 3
    ) -> Optional[List[Any]]:
        """
        Attempt to retrieve the referenced section from the vector store.
        Uses the reference text as a targeted search query.
        """
        ref_text = reference["reference_text"]
        
        # Build a targeted search query from the reference
        search_query = f"content of {ref_text}"
        
        try:
            resolved_chunks = hybrid_search_fn(
                query=search_query, 
                project_id=project_id, 
                k=top_k,
                filter_document_id=reference.get("document_id")
            )
            return resolved_chunks
        except Exception as e:
            print(f"Error resolving reference '{ref_text}': {e}")
            return None
    
    def resolve_all(
        self,
        chunks: List[Any],
        hybrid_search_fn,
        project_id: int,
        max_resolutions: int = 3
    ) -> dict:
        """
        Detect and resolve all cross-references in retrieved chunks.
        Limit to max_resolutions to control latency.
        
        Returns:
            {
                "additional_chunks": List[Any],  # newly retrieved referenced content
                "references_found": int,
                "references_resolved": int,
                "reference_details": List[dict]
            }
        """
        references = self.detect_references(chunks)
        
        if not references:
            return {
                "additional_chunks": [],
                "references_found": 0,
                "references_resolved": 0,
                "reference_details": []
            }
        
        additional_chunks = []
        resolved_count = 0
        
        # Helper to get standard key for comparison
        def get_chunk_key(c) -> str:
            if isinstance(c, tuple):
                doc = c[0]
            else:
                doc = c
            return doc.metadata.get("doc_id_version") or doc.metadata.get("doc_id") or doc.page_content[:100]

        existing_keys = {get_chunk_key(c) for c in chunks}
        
        # Resolve up to max_resolutions references
        for ref in references[:max_resolutions]:
            resolved = self.resolve_reference(
                ref, hybrid_search_fn, project_id
            )
            if resolved:
                new_chunks = []
                for c in resolved:
                    key = get_chunk_key(c)
                    if key not in existing_keys:
                        new_chunks.append(c)
                        existing_keys.add(key)
                
                additional_chunks.extend(new_chunks)
                ref["resolved"] = True
                ref["resolved_chunks"] = [get_chunk_key(c) for c in new_chunks]
                resolved_count += 1
        
        return {
            "additional_chunks": additional_chunks,
            "references_found": len(references),
            "references_resolved": resolved_count,
            "reference_details": references
        }
