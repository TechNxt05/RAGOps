"""
Adaptive Retrieval Router — Paper 4 Core Research Contribution.

Dispatches incoming queries to the retrieval backend actually suited to them,
rather than uniformly applying a heavy hybrid search pipeline:
- Exact/keyword lookup (invoice numbers, product codes, IDs) -> BM25 only
- Structured/aggregation query -> ComputationRouter (math/table aggregation)
- Relationship/multi-hop query -> GraphRAG (stubbed with explicit TODO)
- Semantic/fuzzy query -> FAISS + BM25 + RRF hybrid path
"""

import re
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from app.services.computation_router import ComputationRouter


class RouterDecision(BaseModel):
    query: str
    chosen_backend: str  # "bm25", "computation", "live_database", "graph", "hybrid"
    confidence: float = Field(ge=0.0, le=1.0)
    feature_scores: Dict[str, float]
    fallback_triggered: bool = False
    reasoning: str


# Regex patterns for exact/code/ID lookups
ID_CODE_PATTERNS = [
    re.compile(r'\b(inv|invoice|sku|ticket|id|order|doc|po|ref|num|serial|uuid)[-_\s]?#?\d+\b', re.IGNORECASE),
    re.compile(r'\b[A-Z0-9]{3,8}-[A-Z0-9]{3,8}\b'),
    re.compile(r'\b(code|identifier|version|v)\s*[:=]?\s*[0-9.]+\b', re.IGNORECASE),
    re.compile(r'\b#\d{4,8}\b'),
]

AGGREGATION_KEYWORDS = {
    "total", "sum", "average", "avg", "count", "calculate", "math",
    "percentage", "ratio", "median", "maximum", "max", "minimum", "min",
    "difference between values", "revenue", "cost", "budget", "amount"
}

GRAPH_KEYWORDS = {
    "relationship between", "how is x connected to y", "knowledge graph",
    "multi-hop", "dependency graph", "network of", "entity path",
    "cross-entity lineage", "genealogy", "hierarchy"
}


class AdaptiveRetrievalRouter:
    """
    Query-classification layer that dispatches each incoming query
    to the specific retrieval backend actually suited to it.
    """

    def __init__(self):
        self.computation_router = ComputationRouter()

    def classify(self, query: str, has_external_db: bool = False) -> RouterDecision:
        """
        Classify query into optimal retrieval backend with feature scoring rationale.
        """
        query_str = query.strip()
        query_lower = query_str.lower()
        words = [w for w in re.split(r'\W+', query_lower) if w]

        # 1. Feature scoring
        # Feature A: ID/Code exact match score
        id_pattern_matches = sum(1 for p in ID_CODE_PATTERNS if p.search(query_str))
        id_score = min(1.0, id_pattern_matches * 0.5)
        if len(words) <= 4 and any(c.isdigit() for c in query_str):
            id_score = max(id_score, 0.8)

        # Feature B: Aggregation / Math query score
        agg_hit_count = sum(1 for kw in AGGREGATION_KEYWORDS if kw in query_lower)
        route_info = self.computation_router.route(query_str)
        is_comp_syntax = (route_info.get("route") and getattr(route_info["route"], "value", str(route_info["route"])) == "computation")
        comp_conf = float(route_info.get("confidence", 0.0)) if is_comp_syntax else 0.0
        agg_score = max(min(1.0, agg_hit_count * 0.35), comp_conf if is_comp_syntax else 0.0)

        # Feature C: Graph / Multi-hop relationship score
        graph_hit_count = sum(1 for kw in GRAPH_KEYWORDS if kw in query_lower)
        graph_score = min(1.0, graph_hit_count * 0.45)

        # Feature D: Keyword density score vs natural language flow
        keyword_density = len(set(words)) / float(max(len(words), 1))
        
        feature_scores = {
            "has_id_pattern": round(id_score, 3),
            "has_agg_keywords": round(agg_score, 3),
            "is_multihop_question": round(graph_score, 3),
            "keyword_density": round(keyword_density, 3),
        }

        # 2. Decision Logic
        # Priority 1: Exact ID / Code lookup -> BM25
        if id_score >= 0.7:
            return RouterDecision(
                query=query_str,
                chosen_backend="bm25",
                confidence=id_score,
                feature_scores=feature_scores,
                fallback_triggered=False,
                reasoning=f"High exact ID/code pattern match (score {id_score:.2f}). Dispatched to BM25 index."
            )

        # Priority 2: Aggregation / Calculation -> Connected External DB or Computation backend
        if agg_score >= 0.65:
            chosen = "live_database" if has_external_db else "computation"
            reason = (
                f"Structured aggregation keywords detected (score {agg_score:.2f}). Dispatched to connected External Database."
                if has_external_db else
                f"Structured aggregation keywords detected (score {agg_score:.2f}). Dispatched to ComputationRouter."
            )
            return RouterDecision(
                query=query_str,
                chosen_backend=chosen,
                confidence=agg_score,
                feature_scores=feature_scores,
                fallback_triggered=False,
                reasoning=reason
            )

        # Priority 3: Graph / Multi-hop relationship -> GraphRAG (Stubbed)
        if graph_score >= 0.65:
            return RouterDecision(
                query=query_str,
                chosen_backend="graph",
                confidence=graph_score,
                feature_scores=feature_scores,
                fallback_triggered=False,
                reasoning=f"Cross-entity relationship signals detected (score {graph_score:.2f}). Targeted GraphRAG backend (stubbed -> fallback hybrid)."
            )

        # Default: Hybrid FAISS + BM25 + RRF
        hybrid_confidence = max(0.6, 1.0 - max(id_score, agg_score, graph_score))
        return RouterDecision(
            query=query_str,
            chosen_backend="hybrid",
            confidence=round(hybrid_confidence, 3),
            feature_scores=feature_scores,
            fallback_triggered=False,
            reasoning="Broad natural language query. Dispatched to default Hybrid FAISS + BM25 + RRF pipeline."
        )

    def execute_graph_backend(self, query: str) -> list:
        """
        Executes Stage 1 structure-only document graph walk for relationship/multi-hop queries.
        """
        from app.services.graph_rag_service import graph_rag_service
        return graph_rag_service.search_structural_graph(query)
