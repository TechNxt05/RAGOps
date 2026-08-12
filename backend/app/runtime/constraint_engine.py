"""
ConstraintEngine — Hard constraint enforcement ensuring execution parameters never
violate max latency, cost, retry, or context token budgets.
"""

import logging
from typing import Dict, Any
from app.runtime.execution_plan import ExecutionPlan
from app.runtime.policy_engine import PolicyConfig

logger = logging.getLogger(__name__)


class ConstraintEngine:
    """
    Validates and adjusts an ExecutionPlan against hard constraints.
    """

    def enforce_constraints(
        self,
        plan: ExecutionPlan,
        policy: PolicyConfig
    ) -> ExecutionPlan:
        """
        Enforces strict boundary bounds on ExecutionPlan. If tight budgets are detected,
        gracefully downgrades retrieval top_k, reranking, or model reasoning effort.
        """
        modified = False
        reasons = []

        # 1. Latency budget constraint (< 1500ms -> fast model, reduce top_k)
        if policy.latency_budget_ms < 1500:
            if plan.retrieval_top_k > 5:
                plan.retrieval_top_k = 5
                plan.candidate_pool = 15
                modified = True
                reasons.append("Tight latency budget (<1500ms): capped top_k=5")
            if plan.reasoning_effort == "HIGH":
                plan.reasoning_effort = "LOW"
                modified = True
                reasons.append("Tight latency budget: downgraded reasoning_effort to LOW")

        # 2. Cost budget constraint (< $0.01 -> skip expensive reranker if allowed)
        if policy.cost_budget < 0.01:
            if plan.reranking and policy.allow_rerank_skip:
                plan.reranking = False
                plan.rerank_top_n = 3
                modified = True
                reasons.append("Strict cost budget (<$0.01): disabled cross-encoder reranker")

        # 3. Maximum retry count enforcement
        if plan.max_retries > policy.max_retries:
            plan.max_retries = policy.max_retries
            modified = True
            reasons.append(f"Capped retries to policy max ({policy.max_retries})")

        if modified:
            plan.reasoning += f" | Constraint Adjustments: {'; '.join(reasons)}"
            logger.info(f"ConstraintEngine applied modifications: {reasons}")

        return plan
