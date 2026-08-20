"""
RuntimePlanner — High-level runtime orchestration engine that synthesizes query features,
evaluates policy and constraints, and produces a typed ExecutionPlan BEFORE execution.
"""

import logging
from typing import Dict, Any, Optional
from app.runtime.state import RuntimeState, QueryIntent, CacheStatus
from app.runtime.execution_plan import ExecutionPlan
from app.runtime.policy_engine import PolicyEngine, PolicyPreset, PolicyConfig
from app.runtime.constraint_engine import ConstraintEngine
from app.runtime.decision_log import DecisionLogger, DecisionRecord
from app.runtime.registry import ModelRegistry

logger = logging.getLogger(__name__)


class RuntimePlanner:
    """
    Orchestration layer that sits ABOVE existing service routers (AdaptiveRetrievalRouter, SemanticRouter)
    to set dynamic budgets and execution parameters before pipeline execution.
    """

    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.constraint_engine = ConstraintEngine()
        self.model_registry = ModelRegistry()

    def plan(
        self,
        query: str,
        project_id: int = 1,
        session_id: str = "default_session",
        policy_preset: Optional[PolicyPreset] = None,
        policy_override: Optional[Dict[str, Any]] = None
    ) -> tuple[RuntimeState, ExecutionPlan]:
        """
        Generates RuntimeState and typed ExecutionPlan for an incoming query.
        """
        # 1. Initialize RuntimeState
        state = RuntimeState(
            query=query,
            normalized_query=query.strip().lower(),
            project_id=project_id,
            session_id=session_id
        )

        # 2. Extract query signals (delegated to existing SemanticRouter classification)
        from app.services.semantic_router import SemanticRouter
        sem_router = SemanticRouter()
        sem_res = sem_router.classify(query)
        cat_obj = sem_res.get("category", "factoid")
        cat_name = cat_obj.value if hasattr(cat_obj, "value") else str(cat_obj).lower()

        # Map category to QueryIntent
        intent_map = {
            "factoid": QueryIntent.FACTOID,
            "analytical": QueryIntent.ANALYTICAL,
            "comparative": QueryIntent.COMPARATIVE,
            "procedural": QueryIntent.PROCEDURAL,
            "definitional": QueryIntent.DEFINITIONAL,
        }
        state.intent = intent_map.get(cat_name, QueryIntent.FACTOID)

        # Detect specific feature flags
        state.multi_hop_requirement = any(k in state.normalized_query for k in ["relationship", "lineage", "connected", "across", "between", "multi-hop", "dependency graph"])
        state.mathematical_requirement = any(k in state.normalized_query for k in ["count", "total", "sum", "average", "how many", "math"])
        
        if state.multi_hop_requirement:
            state.intent = QueryIntent.MULTI_HOP
        elif state.mathematical_requirement:
            state.intent = QueryIntent.STRUCTURED

        state.complexity = 0.85 if (state.multi_hop_requirement or state.intent in (QueryIntent.ANALYTICAL, QueryIntent.MULTI_HOP)) else 0.40

        # 3. Resolve PolicyConfig
        policy = self.policy_engine.resolve_policy(preset=policy_preset, custom_override=policy_override)
        state.latency_budget_ms = policy.latency_budget_ms
        state.cost_budget = policy.cost_budget
        state.quality_target = policy.quality_target

        # 3. Synthesize initial ExecutionPlan
        from app.services.adaptive_retrieval_router import AdaptiveRetrievalRouter
        adaptive_router = AdaptiveRetrievalRouter()
        route_decision = adaptive_router.classify(query)
        chosen_backend = route_decision.chosen_backend if hasattr(route_decision, "chosen_backend") else route_decision.get("chosen_backend", "hybrid")

        # Determine adaptive retrieval top_k & reranking
        if state.intent == QueryIntent.FACTOID or chosen_backend == "bm25":
            top_k = 7
            reranking = True
            
            # Use LOW reasoning only for trivial ID lookups, otherwise STANDARD
            is_trivial_lookup = (chosen_backend == "bm25" and len(query.split()) <= 4)
            reasoning_effort = "LOW" if is_trivial_lookup else "STANDARD"
        elif state.intent == QueryIntent.COMPARATIVE:
            top_k = 10
            reranking = True
            reasoning_effort = "STANDARD"
        elif state.intent in (QueryIntent.ANALYTICAL, QueryIntent.MULTI_HOP) or state.multi_hop_requirement:
            top_k = 15
            reranking = True
            reasoning_effort = "HIGH"
        else:
            top_k = 8
            reranking = True
            reasoning_effort = "STANDARD"

        selected_model = self.model_registry.select_best_model(
            max_latency_ms=policy.latency_budget_ms,
            max_cost=policy.cost_budget,
            reasoning_level=reasoning_effort
        )

        plan = ExecutionPlan(
            retrieval_strategy=chosen_backend,
            retrieval_top_k=top_k,
            candidate_pool=max(top_k * 3, 20),
            rerank_top_n=min(top_k, 5),
            reranking=reranking,
            graph_expansion=state.multi_hop_requirement or chosen_backend in ("graph", "graph_rag"),
            query_rewriting=True,
            context_compression=True,
            model=selected_model,
            reasoning_effort=reasoning_effort,
            verification_level=policy.verification_level,
            max_retries=policy.max_retries,
            latency_budget_ms=policy.latency_budget_ms,
            cost_budget=policy.cost_budget,
            reasoning=f"Intent '{state.intent.value}' -> Backend '{chosen_backend}' with top_k={top_k}, model={selected_model}"
        )

        # 5. Enforce Hard Constraints via ConstraintEngine
        plan = self.constraint_engine.enforce_constraints(plan, policy)

        # 6. Attach plan to state trace
        state.selected_retrieval_backend = plan.retrieval_strategy
        state.selected_model = plan.model
        state.execution_plan = plan.to_dict()
        state.add_trace_event("planner_completed", {"plan": plan.to_dict()})

        logger.info(f"RuntimePlanner generated plan for request_id={state.request_id}: {plan.reasoning}")
        return state, plan
