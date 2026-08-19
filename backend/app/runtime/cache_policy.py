"""
CachePolicyEngine — RAGOps Phase 3

Centralized policy decision engine for multi-tier caching:
- Prompt Cache
- Versioned Semantic Cache
- Session Context Cache

Evaluates CachePolicyActions:
- LOOKUP: Check cache before execution
- BYPASS: Force cache bypass for real-time / live queries
- INVALIDATE: Clear cache entries on document ingestion / claim failure
- REFRESH: Re-populate cache after updates
"""

from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel


class CacheAction(str, Enum):
    LOOKUP = "lookup"
    BYPASS = "bypass"
    INVALIDATE = "invalidate"
    REFRESH = "refresh"


class CachePolicyDecision(BaseModel):
    action: CacheAction
    cache_tier: str  # "semantic", "session", "prompt"
    ttl_seconds: int = 3600
    reasoning: str


class CachePolicyEngine:
    """
    Determines cache orchestration strategy based on query intent,
    freshness constraints, and verification status.
    """

    def evaluate_policy(
        self,
        query: str,
        query_intent: str = "factoid",
        force_refresh: bool = False,
        has_unverified_claims: bool = False
    ) -> CachePolicyDecision:
        if force_refresh or has_unverified_claims:
            return CachePolicyDecision(
                action=CacheAction.INVALIDATE if has_unverified_claims else CacheAction.REFRESH,
                cache_tier="semantic",
                ttl_seconds=0,
                reasoning="Bypassing/invalidating cache due to unverified claims or explicit refresh request."
            )

        if query_intent in ("analytical", "multi_hop"):
            return CachePolicyDecision(
                action=CacheAction.BYPASS,
                cache_tier="semantic",
                ttl_seconds=1800,
                reasoning="Complex analytical/multi-hop query requires live pipeline execution."
            )

        return CachePolicyDecision(
            action=CacheAction.LOOKUP,
            cache_tier="semantic",
            ttl_seconds=3600,
            reasoning="Factoid/standard query suitable for semantic cache lookup."
        )


cache_policy_engine = CachePolicyEngine()
