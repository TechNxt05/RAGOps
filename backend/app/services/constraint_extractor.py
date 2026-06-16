import re
from typing import Optional, List
from pydantic import BaseModel

class QueryConstraints(BaseModel):
    # Negation constraints
    excluded_terms: List[str] = []       # "not about X", "excluding X", "except X"
    
    # Date constraints  
    date_after: Optional[str] = None     # "after 2024", "since January"
    date_before: Optional[str] = None    # "before 2023", "until last year"
    
    # Document type constraints
    doc_types: List[str] = []            # "in PDF", "from Word documents"
    
    # Source constraints
    source_contains: List[str] = []      # "from the HR policy", "in the contract"
    
    # Numeric constraints
    numeric_filters: List[dict] = []     # [{"field": "price", "op": "lt", "value": 200}]
    
    has_constraints: bool = False

class ConstraintExtractor:
    def extract(self, query: str) -> QueryConstraints:
        constraints = QueryConstraints()
        query_lower = query.lower()
        
        # Negation extraction
        neg_patterns = [
            r"not (?:about|related to|regarding|concerning) ([a-z\s]+?)(?:\s+and|\s+or|$|,|\?)",
            r"excluding ([a-z\s]+?)(?:\s+and|\s+or|$|,|\?)",
            r"except (?:for )?([a-z\s]+?)(?:\s+and|\s+or|$|,|\?)",
            r"without ([a-z\s]+?)(?:\s+and|\s+or|$|,|\?)",
        ]
        for pattern in neg_patterns:
            matches = re.findall(pattern, query_lower)
            constraints.excluded_terms.extend([m.strip() for m in matches if m.strip()])
        
        # Date extraction
        date_after = re.search(
            r"(?:after|since|from) (\d{4}|january|february|march|april|may|june|"
            r"july|august|september|october|november|december)", query_lower
        )
        if date_after:
            constraints.date_after = date_after.group(1)
        
        date_before = re.search(
            r"(?:before|until|up to) (\d{4}|january|february|march|april|may|june|"
            r"july|august|september|october|november|december)", query_lower
        )
        if date_before:
            constraints.date_before = date_before.group(1)
        
        # Source constraints
        source_patterns = [
            r"(?:from|in) the ([a-z\s]+?) (?:document|policy|contract|file|report)",
            r"(?:from|in) ([a-z\s]+?)(?:\s+document|\s+file|\s+policy|$)",
        ]
        for pattern in source_patterns:
            matches = re.findall(pattern, query_lower)
            constraints.source_contains.extend([m.strip() for m in matches if m.strip()])
        
        # Document types
        if re.search(r"in pdf", query_lower):
            constraints.doc_types.append("pdf")
            
        # Set has_constraints flag
        constraints.has_constraints = any([
            constraints.excluded_terms,
            constraints.date_after,
            constraints.date_before,
            constraints.doc_types,
            constraints.source_contains,
            constraints.numeric_filters,
        ])
        
        return constraints
    
    def apply_to_chunks(
        self, 
        chunks: List[dict], 
        constraints: QueryConstraints
    ) -> List[dict]:
        """
        Post-filter retrieved chunks against hard constraints.
        Called AFTER retrieval to enforce constraints the vector DB
        cannot handle natively.
        """
        if not constraints.has_constraints:
            return chunks
        
        filtered = chunks
        
        # Apply exclusion filters
        for excluded_term in constraints.excluded_terms:
            filtered = [
                c for c in filtered 
                if excluded_term.lower() not in c.get("text", c.get("content", "")).lower()
            ]
        
        # Apply source filters
        for source in constraints.source_contains:
            source_filtered = [
                c for c in filtered
                if source.lower() in str(c.get("source", "")).lower() or
                   source.lower() in c.get("text", c.get("content", "")).lower()
            ]
            if source_filtered:  # Only apply if it doesn't eliminate everything
                filtered = source_filtered
                
        # Apply doc_types filter
        for doc_type in constraints.doc_types:
            type_filtered = [
                c for c in filtered
                if doc_type.lower() in str(c.get("source", "")).lower()
            ]
            if type_filtered:
                filtered = type_filtered
        
        return filtered
