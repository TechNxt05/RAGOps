import re
from enum import Enum
from typing import Optional

class QueryRoute(Enum):
    COMPUTATION = "computation"
    RETRIEVAL = "retrieval"

class ComputationRouter:
    # Tier 1: Aggregation verbs - always route to computation
    TIER1_PATTERNS = [
        r"\b(total|sum|count|how many|average|mean|maximum|minimum|highest|lowest|"
        r"percentage|percent|ratio|most|least|top \d+|bottom \d+)\b",
    ]
    
    # Tier 2: Numeric comparisons - route to computation
    TIER2_PATTERNS = [
        r"\b(greater than|less than|more than|fewer than|above|below|"
        r"over|under|at least|at most|between .* and)\b",
        r"(>\s*\d+|<\s*\d+|\>=\s*\d+|\<=\s*\d+)",
    ]
    
    # Tier 3: Retrieval signals - route to retrieval
    TIER3_PATTERNS = [
        r"^(find|show|list|fetch|get|display|what is|who is|where is|"
        r"tell me about|describe|explain)",
    ]
    
    def route(self, query: str) -> dict:
        query_lower = query.lower().strip()
        
        # Tier 1 check
        for pattern in self.TIER1_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                return {
                    "route": QueryRoute.COMPUTATION,
                    "tier": 1,
                    "matched_signal": match.group(),
                    "confidence": 0.97
                }
        
        # Tier 2 check
        for pattern in self.TIER2_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                return {
                    "route": QueryRoute.COMPUTATION,
                    "tier": 2,
                    "matched_signal": match.group(),
                    "confidence": 0.90
                }
        
        # Tier 3 check
        for pattern in self.TIER3_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                return {
                    "route": QueryRoute.RETRIEVAL,
                    "tier": 3,
                    "matched_signal": match.group(),
                    "confidence": 0.85
                }
        
        # Default: RETRIEVAL (safer for RAG context)
        return {
            "route": QueryRoute.RETRIEVAL,
            "tier": 0,
            "matched_signal": None,
            "confidence": 0.60
        }
    
    def compute_aggregation(self, query: str, project_id: int, db) -> Optional[dict]:
        """
        For COMPUTATION-routed queries, attempt to answer from structured
        document metadata in PostgreSQL rather than vector search.
        
        Returns computed answer dict or None if computation not possible.
        """
        query_lower = query.lower()
        
        # Document count
        if re.search(r"how many (documents|files|pdfs)", query_lower):
            from sqlmodel import select, func
            from app.models.rag import Document
            
            # Query document count for this project
            count_statement = select(func.count(Document.id)).where(Document.project_id == project_id)
            count = db.exec(count_statement).one()
            
            return {
                "answer_type": "computation",
                "computation_method": "sql_aggregate",
                "note": "Answered from document metadata, not vector search",
                "answer": f"There are {count} documents in this project."
            }
        
        # Cannot do full-scan aggregation on unstructured text
        # Return None to fall back to retrieval with warning
        return None
