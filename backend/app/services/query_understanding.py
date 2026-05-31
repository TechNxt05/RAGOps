"""
Query Understanding Stage — pre-retrieval query analysis.
Classifies complexity, decomposes multi-part queries, expands ambiguous terms.
Sits BEFORE the hybrid retrieval stage.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import re

class QueryComplexity(str, Enum):
    FACTOID = "factoid"        # "What is RAG?" → single fact lookup
    ANALYTICAL = "analytical"  # "How does X compare to Y?" → reasoning required
    MULTI_HOP = "multi_hop"   # "Why did X cause Y which led to Z?" → chain reasoning

@dataclass
class QueryAnalysis:
    original_query: str
    complexity: QueryComplexity
    sub_queries: List[str]          # Decomposed for multi-hop
    expanded_query: str             # Enriched with synonyms/context
    retrieval_strategy: str         # "single" or "multi" (affects top-k)
    suggested_top_k: int            # Based on complexity
    confidence: float               # 0-1 confidence in classification

FACTOID_PATTERNS = [
    re.compile(r'^(what is|what are|who is|where is|when did|what does)\b', re.I),
    re.compile(r'^(define|definition of|meaning of)\b', re.I),
    re.compile(r'^(list|name|give me)\b.{0,40}$', re.I),
    re.compile(r'^(how many|how much)\b', re.I),
]

ANALYTICAL_KEYWORDS = frozenset({
    'compare', 'contrast', 'difference', 'versus', 'vs',
    'advantage', 'disadvantage', 'tradeoff', 'trade-off',
    'why', 'explain', 'how does', 'what happens when',
    'analyze', 'evaluate', 'assess', 'impact', 'effect',
})

MULTI_HOP_KEYWORDS = frozenset({
    'led to', 'caused by', 'resulted in', 'chain', 'sequence',
    'first', 'then', 'finally', 'relationship between',
    'how does x affect y', 'what is the connection',
})

class QueryUnderstanding:
    """
    Pre-retrieval query analysis pipeline.
    Runs in <5ms — no LLM calls, pure heuristic classification.
    """
    
    def analyze(self, query: str) -> QueryAnalysis:
        query = query.strip()
        complexity, confidence = self._classify_complexity(query)
        sub_queries = self._decompose(query, complexity)
        expanded = self._expand_query(query, complexity)
        
        # Adjust retrieval parameters based on complexity
        strategy_map = {
            QueryComplexity.FACTOID: ("single", 5),
            QueryComplexity.ANALYTICAL: ("single", 10),
            QueryComplexity.MULTI_HOP: ("multi", 15),
        }
        retrieval_strategy, top_k = strategy_map[complexity]
        
        return QueryAnalysis(
            original_query=query,
            complexity=complexity,
            sub_queries=sub_queries,
            expanded_query=expanded,
            retrieval_strategy=retrieval_strategy,
            suggested_top_k=top_k,
            confidence=confidence,
        )
    
    def _classify_complexity(self, query: str) -> tuple[QueryComplexity, float]:
        q_lower = query.lower()
        
        # Factoid check first (fast path)
        for pattern in FACTOID_PATTERNS:
            if pattern.match(query):
                return QueryComplexity.FACTOID, 0.95
        
        # Multi-hop check
        multi_hop_hits = sum(1 for kw in MULTI_HOP_KEYWORDS if kw in q_lower)
        if multi_hop_hits >= 2:
            return QueryComplexity.MULTI_HOP, min(0.7 + multi_hop_hits * 0.1, 0.95)
        
        # Analytical check
        analytical_hits = sum(1 for kw in ANALYTICAL_KEYWORDS if kw in q_lower)
        if analytical_hits >= 1:
            return QueryComplexity.ANALYTICAL, min(0.6 + analytical_hits * 0.15, 0.95)
        
        # Default: treat as factoid if short, analytical if long
        if len(query.split()) <= 8:
            return QueryComplexity.FACTOID, 0.6
        return QueryComplexity.ANALYTICAL, 0.55
    
    def _decompose(self, query: str, complexity: QueryComplexity) -> List[str]:
        """Decompose multi-hop queries into sub-queries for parallel retrieval"""
        if complexity != QueryComplexity.MULTI_HOP:
            return [query]
        
        # Split on reasoning connectors
        parts = re.split(
            r'\b(and also|furthermore|additionally|moreover|first|then|finally|'
            r'as well as|in addition|subsequently)\b',
            query, flags=re.I
        )
        sub_queries = [p.strip() for p in parts if len(p.strip()) > 15]
        return sub_queries if len(sub_queries) > 1 else [query]
    
    def _expand_query(self, query: str, complexity: QueryComplexity) -> str:
        """
        Light query expansion — add domain context.
        For factoid queries: return as-is (expansion hurts precision).
        For analytical: append context hint.
        """
        if complexity == QueryComplexity.FACTOID:
            return query
        
        # For analytical/multi-hop, append a retrieval hint
        # (signals to BM25 that this is a comparative query)
        expansion_hints = {
            QueryComplexity.ANALYTICAL: "comparison analysis explanation",
            QueryComplexity.MULTI_HOP: "relationship cause effect sequence",
        }
        hint = expansion_hints.get(complexity, "")
        return f"{query} {hint}".strip() if hint else query


def get_query_understanding() -> QueryUnderstanding:
    return QueryUnderstanding()
