import re
from typing import List, Optional, Dict, Any
from itertools import combinations

class ConflictDetector:
    # Signals that often indicate factual claims that can conflict
    FACTUAL_SIGNALS = [
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",  # dates e.g. 12/31/2024 or 12-31-2024
        r"\b((?:19|20)\d{2})\b",                  # years
        r"\$[\d,]+(?:\.\d{2})?",                 # money amounts e.g. $1,000.50
        r"\b(\d+(?:\.\d+)?)\s*(?:percent|%)\b",   # percentages
        r"\b(\d+)\s*(?:days|months|years|hours|weeks)\b",  # durations
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",  # month day year
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",  # month+year
    ]
    
    def detect_conflicts(
        self, 
        chunks: List[Any],
        query: str
    ) -> dict:
        """
        Check retrieved chunks for factual contradictions.
        
        Returns:
            {
                "has_conflicts": bool,
                "conflict_pairs": List[dict],
                "conflict_severity": "none" | "low" | "high",
                "conflicting_chunk_ids": List[str]
            }
        """
        parsed_chunks = []
        for i, item in enumerate(chunks):
            if isinstance(item, tuple):
                doc = item[0]
                parsed_chunks.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", f"document_{i}"),
                    "id": str(doc.metadata.get("doc_id_version") or doc.metadata.get("doc_id") or i)
                })
            elif hasattr(item, "page_content"):
                parsed_chunks.append({
                    "content": item.page_content,
                    "source": item.metadata.get("source", f"document_{i}"),
                    "id": str(item.metadata.get("doc_id_version") or item.metadata.get("doc_id") or i)
                })
            elif isinstance(item, dict):
                parsed_chunks.append({
                    "content": item.get("content") or item.get("text") or "",
                    "source": item.get("source") or f"chunk_{i}",
                    "id": str(item.get("id") or item.get("doc_id_version") or i)
                })

        if len(parsed_chunks) < 2:
            return {
                "has_conflicts": False,
                "conflict_pairs": [],
                "conflict_severity": "none",
                "conflicting_chunk_ids": []
            }
        
        conflict_pairs = []
        
        # Check all chunk pairs for contradictions
        for i, j in combinations(range(len(parsed_chunks)), 2):
            chunk_a = parsed_chunks[i]
            chunk_b = parsed_chunks[j]
            
            conflicts = self._find_value_conflicts(
                chunk_a["content"],
                chunk_b["content"],
                query
            )
            
            if conflicts:
                conflict_pairs.append({
                    "chunk_a_index": i,
                    "chunk_b_index": j,
                    "chunk_a_source": chunk_a["source"],
                    "chunk_b_source": chunk_b["source"],
                    "conflicting_values": conflicts,
                    "conflict_type": self._classify_conflict(conflicts)
                })
        
        conflicting_ids = list(set(
            [parsed_chunks[p["chunk_a_index"]]["id"] for p in conflict_pairs] +
            [parsed_chunks[p["chunk_b_index"]]["id"] for p in conflict_pairs]
        ))
        
        severity = "none"
        if conflict_pairs:
            severity = "high" if len(conflict_pairs) > 2 else "low"
        
        return {
            "has_conflicts": len(conflict_pairs) > 0,
            "conflict_pairs": conflict_pairs,
            "conflict_severity": severity,
            "conflicting_chunk_ids": conflicting_ids
        }
    
    def _find_value_conflicts(
        self, 
        text_a: str, 
        text_b: str,
        query: str
    ) -> List[dict]:
        """
        Extract factual values from both texts and compare.
        Returns list of conflicting value pairs.
        """
        conflicts = []
        
        for pattern in self.FACTUAL_SIGNALS:
            values_a = set(v.lower().strip() for v in re.findall(pattern, text_a, re.IGNORECASE) if isinstance(v, str) and v.strip())
            values_b = set(v.lower().strip() for v in re.findall(pattern, text_b, re.IGNORECASE) if isinstance(v, str) and v.strip())
            
            # If both chunks mention the same type of fact but different values
            if values_a and values_b and values_a != values_b:
                conflicts.append({
                    "pattern_type": pattern,
                    "values_in_chunk_a": list(values_a),
                    "values_in_chunk_b": list(values_b)
                })
        
        return conflicts
    
    def _classify_conflict(self, conflicts: List[dict]) -> str:
        """Classify the type of conflict for display."""
        for conflict in conflicts:
            pattern = conflict["pattern_type"]
            if "date" in pattern or "january" in pattern:
                return "date_conflict"
            if r"\$" in pattern:
                return "amount_conflict"
            if "percent" in pattern:
                return "percentage_conflict"
            if "days|months" in pattern:
                return "duration_conflict"
        return "value_conflict"
    
    def build_conflict_prompt_addendum(
        self, 
        conflict_result: dict
    ) -> Optional[str]:
        """
        If conflicts detected, add instructions to LLM generation prompt
        telling it to acknowledge the conflict instead of silently picking one.
        """
        if not conflict_result["has_conflicts"]:
            return None
        
        pairs = conflict_result["conflict_pairs"]
        addendum = "\n\nIMPORTANT: The retrieved documents contain conflicting information:\n"
        
        for pair in pairs[:3]:  # Show max 3 conflicts
            addendum += (
                f"- '{pair['chunk_a_source']}' and '{pair['chunk_b_source']}' "
                f"contain conflicting {pair['conflict_type'].replace('_', ' ')}.\n"
            )
        
        addendum += (
            "\nDo NOT silently pick one value. Explicitly acknowledge the conflict "
            "in your response and cite both sources. Tell the user which document "
            "each value comes from."
        )
        
        return addendum
