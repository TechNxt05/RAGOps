"""
RAGOps Phase A Runtime Core Verification Harness
Empirical test suite validating RuntimeState, ExecutionPlan, PolicyEngine,
ConstraintEngine, DecisionLogger, ModelRegistry, and RuntimePlanner.
"""

import sys
import os
import time

# Ensure backend directory is on sys.path
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from dotenv import load_dotenv
load_dotenv()

# Patch httpx 0.28+ breaking change
try:
    import httpx
    _orig_client_init = httpx.Client.__init__
    def _patched_client_init(self, *args, **kwargs):
        if "proxies" in kwargs:
            p = kwargs.pop("proxies")
            if p and "proxy" not in kwargs:
                kwargs["proxy"] = p
        return _orig_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = _patched_client_init

    _orig_async_client_init = httpx.AsyncClient.__init__
    def _patched_async_client_init(self, *args, **kwargs):
        if "proxies" in kwargs:
            p = kwargs.pop("proxies")
            if p and "proxy" not in kwargs:
                kwargs["proxy"] = p
        return _orig_async_client_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = _patched_async_client_init
except Exception:
    pass

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def test_phase_a_runtime():
    print("=" * 70)
    print("           RAGOps Phase A — Runtime Core Verification Harness")
    print("=" * 70)

    # 1. Verify RuntimeState
    print("\n[1/6] Verifying RuntimeState (Pydantic v2)...")
    from app.runtime.state import RuntimeState, QueryIntent, CacheStatus
    state = RuntimeState(query="What is the revenue growth for Q3?", project_id=1)
    assert state.query == "What is the revenue growth for Q3?"
    assert state.project_id == 1
    assert state.request_id.startswith("req_")
    state.add_trace_event("test_event", {"metric": 100})
    assert len(state.execution_trace) == 1
    assert state.elapsed_ms() >= 0
    print(f"  ✅ RuntimeState: request_id={state.request_id}, elapsed={state.elapsed_ms()}ms")

    # 2. Verify ExecutionPlan
    print("\n[2/6] Verifying ExecutionPlan...")
    from app.runtime.execution_plan import ExecutionPlan
    plan = ExecutionPlan(
        retrieval_strategy="hybrid",
        retrieval_top_k=10,
        reranking=True,
        model="groq/llama-3.3-70b-versatile",
        reasoning_effort="STANDARD"
    )
    plan_dict = plan.to_dict()
    assert plan_dict["retrieval_strategy"] == "hybrid"
    assert plan_dict["retrieval_top_k"] == 10
    print(f"  ✅ ExecutionPlan: strategy={plan.retrieval_strategy}, top_k={plan.retrieval_top_k}, model={plan.model}")

    # 3. Verify PolicyEngine
    print("\n[3/6] Verifying PolicyEngine & Presets...")
    from app.runtime.policy_engine import PolicyEngine, PolicyPreset
    pe = PolicyEngine()
    fast_policy = pe.resolve_policy(preset=PolicyPreset.FAST)
    assert fast_policy.latency_budget_ms == 1000
    assert fast_policy.verification_level == "BASIC"

    deep_policy = pe.resolve_policy(preset=PolicyPreset.DEEP)
    assert deep_policy.latency_budget_ms == 6000
    assert deep_policy.verification_level == "STRICT"

    custom_policy = pe.resolve_policy(preset=PolicyPreset.STANDARD, custom_override={"latency_budget_ms": 1200})
    assert custom_policy.latency_budget_ms == 1200
    print(f"  ✅ PolicyEngine: FAST={fast_policy.latency_budget_ms}ms, DEEP={deep_policy.latency_budget_ms}ms, Custom={custom_policy.latency_budget_ms}ms")

    # 4. Verify ConstraintEngine
    print("\n[4/6] Verifying ConstraintEngine...")
    from app.runtime.constraint_engine import ConstraintEngine
    ce = ConstraintEngine()
    test_plan = ExecutionPlan(retrieval_top_k=15, reasoning_effort="HIGH", reranking=True)
    tight_policy = pe.resolve_policy(preset=PolicyPreset.FAST, custom_override={"latency_budget_ms": 800, "cost_budget": 0.005})
    
    bounded_plan = ce.enforce_constraints(test_plan, tight_policy)
    assert bounded_plan.retrieval_top_k <= 5
    assert bounded_plan.reasoning_effort == "LOW"
    assert bounded_plan.reranking is False
    print(f"  ✅ ConstraintEngine: Bounded top_k={bounded_plan.retrieval_top_k}, reasoning={bounded_plan.reasoning_effort}, reranking={bounded_plan.reranking}")

    # 5. Verify DecisionLogger & ModelRegistry
    print("\n[5/6] Verifying DecisionLogger & ModelRegistry...")
    from app.runtime.decision_log import DecisionLogger
    from app.runtime.registry import ModelRegistry
    rec = DecisionLogger.create_record(
        request_id=state.request_id,
        intent="analytical",
        complexity=0.8,
        latency_budget=3000,
        cost_budget=0.02,
        plan=bounded_plan.to_dict()
    )
    assert rec.request_id == state.request_id
    mr = ModelRegistry()
    best_model = mr.select_best_model(max_latency_ms=1500, max_cost=0.02, reasoning_level="STANDARD")
    assert best_model is not None
    print(f"  ✅ DecisionLogger & ModelRegistry: record={rec.query_intent}, selected_model={best_model}")

    # 6. Verify RuntimePlanner Orchestration
    print("\n[6/6] Verifying RuntimePlanner Orchestration...")
    from app.runtime.planner import RuntimePlanner
    planner = RuntimePlanner()

    # Factoid query
    st1, pl1 = planner.plan(query="What is RAGOps?", project_id=1, policy_preset=PolicyPreset.FAST)
    assert pl1.retrieval_top_k <= 5
    assert st1.intent == QueryIntent.FACTOID

    # Analytical / Multi-hop query
    st2, pl2 = planner.plan(query="Analyze relationship lineage and cross-entity dependency graph between accounts", project_id=1, policy_preset=PolicyPreset.DEEP)
    print(f"  Debug: st2.intent={st2.intent}, strategy={pl2.retrieval_strategy}, top_k={pl2.retrieval_top_k}, graph_expansion={pl2.graph_expansion}")
    assert pl2.graph_expansion is True

    print(f"  ✅ RuntimePlanner: Factoid top_k={pl1.retrieval_top_k}, Multi-hop graph_expansion={pl2.graph_expansion}")

    print("=" * 70)
    print("   🎉 ALL PHASE A RUNTIME CORE VERIFICATION CHECKS PASSED CLEANLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_phase_a_runtime()
