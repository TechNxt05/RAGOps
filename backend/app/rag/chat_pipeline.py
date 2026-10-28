"""
Chat turn pipeline.

    resolve session/access -> semantic cache -> query understanding/rewrite ->
    retrieval (ONCE) -> confidence gate (+ self-correction / absence proof) ->
    generation (one LLM call, streamable, model fallback chain) ->
    evaluation (RAGAS-style metrics, output contract, claim verification) ->
    persistence (message, query log, token usage, caches)

The chunks retrieved here are exactly the chunks placed in the prompt, cited in
the answer, shown as sources, and used for every quality metric.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from fastapi import HTTPException
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from starlette.concurrency import run_in_threadpool
from sqlmodel import Session, select

from app.auth.access import require_project
from app.core.config import settings
from app.core.rate_limit import rate_limiter
from app.db import engine
from app.models.chat import ChatSession, Message
from app.models.query_log import QueryLog
from app.models.rag import Project, RAGConfig
from app.models.usage import TokenUsage
from app.models.user import User
from app.rag.engine import RAGEngine, SearchResultList
from app.services import llm_factory, semantic_cache
from app.services.cost_control import get_cost_manager
from app.services.rag_evaluator import evaluate_rag_response
from app.utils.cost import calculate_cost
from app.utils.llm_response import extract_delta, extract_text

logger = logging.getLogger(__name__)

_background_tasks: set = set()

NOT_FOUND_MARKERS = (
    "not found", "cannot find", "can't find", "no information", "do not have information",
    "does not mention", "not mentioned", "don't have info", "no mention", "does not contain",
)


@dataclass
class Turn:
    req: Any
    user: User
    db: Session
    t0: float = field(default_factory=time.time)
    chat_session: Optional[ChatSession] = None
    is_new_session: bool = False
    project_id: Optional[int] = None
    config: Optional[RAGConfig] = None
    kb_version: int = 1
    history: List[Any] = field(default_factory=list)
    history_dicts: List[dict] = field(default_factory=list)
    related_context: str = ""
    related_titles: List[str] = field(default_factory=list)
    chain: List[Dict[str, str]] = field(default_factory=list)
    utility_llm: Any = None
    engine: Optional[RAGEngine] = None
    results: List[Tuple[LCDocument, float]] = field(default_factory=list)
    gate_result: Any = None
    trace: Dict[str, Any] = field(default_factory=dict)
    search_stats: Dict[str, Any] = field(default_factory=dict)
    conflict_res: Any = None
    query_analysis: Any = None
    not_found_proof: Optional[str] = None
    retrieval_query: str = ""
    turn_type: str = "fresh_retrieval"
    cacheable: bool = False
    adaptive_backend: str = "hybrid"
    tier: str = "standard"
    timings: Dict[str, float] = field(default_factory=dict)
    _mark: float = field(default_factory=time.time)

    def lap(self, phase: str) -> None:
        now = time.time()
        self.timings[phase] = round((now - self._mark) * 1000.0, 1)
        self._mark = now


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

def _resolve_session(turn: Turn) -> None:
    req, db, user = turn.req, turn.db, turn.user
    if req.session_id:
        cs = db.get(ChatSession, req.session_id)
        if not cs or cs.user_id != user.id:
            raise HTTPException(status_code=404, detail="Session not found")
        turn.chat_session = cs
        turn.project_id = cs.project_id
    else:
        if not req.project_id:
            raise HTTPException(status_code=400, detail="project_id is required for new session")
        turn.project_id = req.project_id
    require_project(db, user, turn.project_id)

    if turn.chat_session is None:
        turn.chat_session = ChatSession(user_id=user.id, title=req.title or req.content[:30], project_id=turn.project_id)
        turn.is_new_session = True
    turn.chat_session.settings = {
        "model_provider": req.model_provider,
        "model_name": req.model_name,
        "temperature": req.temperature,
        "history_limit": req.history_limit,
        "project_context_limit": req.project_context_limit,
    }
    db.add(turn.chat_session)
    db.commit()
    db.refresh(turn.chat_session)


def _load_history(turn: Turn, exclude_id: int) -> None:
    db, req = turn.db, turn.req
    limit = max(0, min(int(req.history_limit or 0), 20))
    past = db.exec(
        select(Message)
        .where(Message.session_id == turn.chat_session.id)
        .where(Message.id != exclude_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    ).all() if limit else []
    past = sorted(past, key=lambda m: m.created_at)
    for msg in past:
        if msg.role == "user":
            turn.history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            turn.history.append(AIMessage(content=msg.content))
    turn.history_dicts = [{"role": m.type, "content": m.content} for m in turn.history]

    # Related chats: ONLY the current user's own sessions (never other users').
    ids = [i for i in (req.context_session_ids or []) if i != turn.chat_session.id]
    q = select(ChatSession).where(ChatSession.user_id == turn.user.id).where(ChatSession.id != turn.chat_session.id)
    if ids:
        sessions = db.exec(q.where(ChatSession.id.in_(ids))).all()
    elif (req.project_context_limit or 0) > 0:
        sessions = db.exec(
            q.where(ChatSession.project_id == turn.project_id)
            .order_by(ChatSession.created_at.desc())
            .limit(min(int(req.project_context_limit), 5))
        ).all()
    else:
        sessions = []
    parts = []
    for s in sessions:
        msgs = db.exec(
            select(Message).where(Message.session_id == s.id).order_by(Message.created_at.desc()).limit(3)
        ).all()
        if msgs:
            turn.related_titles.append(s.title or f"Chat {s.id}")
            lines = [f"  {m.role.upper()}: {m.content[:200]}" for m in sorted(msgs, key=lambda m: m.created_at)]
            parts.append(f"- Chat '{s.title}':\n" + "\n".join(lines))
    if parts:
        turn.related_context = "Your earlier conversations in this project (for continuity only):\n" + "\n".join(parts)


def _build_chain(turn: Turn) -> None:
    req, config = turn.req, turn.config
    preferred = None
    if req.model_name and (req.model_provider or "").lower() in llm_factory.SUPPORTED_PROVIDERS:
        preferred = {"provider": req.model_provider.lower(), "name": req.model_name}

    routing = get_cost_manager().router.route(req.content)
    turn.tier = routing.tier.value
    chain = llm_factory.model_chain(config, preferred)
    # Optional cost routing: simple questions go to the project's fallback (cheaper) model first.
    if getattr(config, "enable_cost_routing", False) and turn.tier == "simple" and config.fallback_models:
        cheap = [m for m in chain if m in [{"provider": f.get("provider"), "name": f.get("name")} for f in config.fallback_models]]
        chain = cheap + [m for m in chain if m not in cheap]
    turn.chain = chain
    turn.utility_llm, _ = llm_factory.build_with_fallbacks(chain, temperature=0.0, max_tokens=2048, fast=True)


def prepare(req, db: Session, user: User) -> Turn:
    turn = Turn(req=req, user=user, db=db)
    rate_limiter.check(f"chat:{user.id}", settings.rate_limit_chat_per_min)
    if not (req.content or "").strip():
        raise HTTPException(status_code=400, detail="Message is empty")
    _resolve_session(turn)

    user_msg = Message(session_id=turn.chat_session.id, role="user", content=req.content)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    project = db.get(Project, turn.project_id)
    turn.kb_version = (project.kb_version or 1) if project else 1
    turn.engine = RAGEngine(db)
    try:
        turn.config = turn.engine.get_active_config(turn.project_id)
    except ValueError:
        turn.config = RAGConfig(project_id=turn.project_id)
    _load_history(turn, exclude_id=user_msg.id)
    _build_chain(turn)
    return turn


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _to_results(docs) -> List[Tuple[LCDocument, float]]:
    return [(d, float(s)) for d, s in docs]


async def retrieve(turn: Turn) -> None:
    from app.services.adaptive_retrieval_router import AdaptiveRetrievalRouter, RouterDecision
    from app.services.computation_router import ComputationRouter, QueryRoute
    from app.services.constraint_extractor import ConstraintExtractor
    from app.services.query_rewriter import QueryRewriter
    from app.services.session_context_cache import session_cache
    from app.services.turn_type_router import TurnType, TurnTypeRouter

    content, config, sid = turn.req.content, turn.config, turn.chat_session.id

    if getattr(config, "enable_adaptive_router", True):
        decision = AdaptiveRetrievalRouter().classify(content)
    else:
        decision = RouterDecision(query=content, chosen_backend="hybrid", confidence=1.0,
                                  feature_scores={}, fallback_triggered=False,
                                  reasoning="Adaptive router disabled for this project.")
    if decision.chosen_backend == "graph":
        decision.fallback_triggered = True
        decision.chosen_backend = "hybrid"
        decision.reasoning += " [GraphRAG backend not enabled; using hybrid search]"
    turn.adaptive_backend = decision.chosen_backend

    has_context = session_cache.has_context(sid)
    turn_type = TurnTypeRouter().classify(content, session_has_context=has_context, conversation_history=turn.history_dicts)
    turn.turn_type = turn_type.value if hasattr(turn_type, "value") else str(turn_type)

    rewrite_info: Dict[str, Any] = {"was_rewritten": False}
    route_decision: Dict[str, Any] = {"route": "retrieval", "tier": 0, "matched_signal": None, "confidence": 0.6}
    constraints_applied: Dict[str, Any] = {"has_constraints": False}
    retrieval_source = "skipped"
    results: List[Tuple[LCDocument, float]] = SearchResultList()

    if turn_type == TurnType.CHIT_CHAT:
        pass
    else:
        comp = ComputationRouter()
        route_decision = (
            {"route": QueryRoute.COMPUTATION, "tier": 1, "matched_signal": "adaptive_router_computation", "confidence": decision.confidence}
            if decision.chosen_backend == "computation" else comp.route(content)
        )
        comp_answer = None
        if route_decision["route"] == QueryRoute.COMPUTATION:
            comp_answer = await run_in_threadpool(comp.compute_aggregation, content, turn.project_id, turn.db)
        if comp_answer:
            results = SearchResultList([(LCDocument(page_content=comp_answer["answer"],
                                                    metadata={"source": "Database Computation", "doc_id": 0}), 1.0)])
            retrieval_source = "computation_engine"
        else:
            route_decision["route"] = "retrieval"
            query = content
            if getattr(config, "enable_query_rewrite", True) and turn.history_dicts:
                rewriter = QueryRewriter(turn.utility_llm)
                if turn_type == TurnType.FOLLOW_UP:
                    # The turn router already decided this depends on history; rewrite unconditionally.
                    rewriter._needs_rewriting = lambda _q: True
                rewrite_info = await rewriter.rewrite(content, turn.history_dicts)
                query = rewrite_info["rewritten_query"] or content
            turn.retrieval_query = query
            # Only standalone questions are cacheable (answers to follow-ups depend on history).
            turn.cacheable = not rewrite_info.get("was_rewritten") and turn_type != TurnType.FOLLOW_UP

            constraints = ConstraintExtractor().extract(query)
            constraints_applied = {
                "has_constraints": constraints.has_constraints,
                "excluded_terms": constraints.excluded_terms,
                "date_after": constraints.date_after,
                "date_before": constraints.date_before,
                "doc_types": constraints.doc_types,
                "source_contains": constraints.source_contains,
            }
            results = await run_in_threadpool(
                turn.engine.search,
                query=query, project_id=turn.project_id, k=config.top_k,
                score_threshold=config.similarity_threshold, constraints=constraints,
                rewritten_query=query if rewrite_info.get("was_rewritten") else None,
                llm_client=turn.utility_llm,
            )
            retrieval_source = "vector_db"
            if not results and turn_type == TurnType.FOLLOW_UP:
                cached = session_cache.get(sid) or []
                results = SearchResultList([(LCDocument(page_content=c["text"], metadata={"source": c.get("source"), "doc_id": c.get("id")}), 1.0) for c in cached])
                retrieval_source = "session_cache"
            gate = getattr(results, "confidence_gate_result", None)
            if gate and gate.passed:
                session_cache.store(sid, query, [
                    {"id": str(d.metadata.get("doc_id", "")), "text": d.page_content, "source": d.metadata.get("source", "")}
                    for d, _ in results
                ])

    trace = getattr(results, "pipeline_trace", None) or {"query": content, "status": "success", "stages": {}}
    trace["adaptive_router_decision"] = decision.model_dump() if hasattr(decision, "model_dump") else decision.dict()
    trace["turn_type"] = turn.turn_type
    trace["retrieval_source"] = retrieval_source
    trace["rewrite_info"] = rewrite_info
    trace["route_decision"] = {
        "route": getattr(route_decision["route"], "value", str(route_decision["route"])),
        "tier": route_decision.get("tier"),
        "matched_signal": route_decision.get("matched_signal"),
        "confidence": route_decision.get("confidence"),
    }
    trace["constraints_applied"] = constraints_applied
    trace["model_tier"] = turn.tier
    trace["model_chain"] = turn.chain[:4]

    turn.trace = trace
    turn.results = _to_results(results)
    turn.gate_result = getattr(results, "confidence_gate_result", None)
    turn.conflict_res = getattr(results, "conflict_detection", None)
    turn.query_analysis = getattr(results, "query_analysis", None)
    turn.search_stats = {
        "chunks_before_pruning": getattr(results, "chunks_before_pruning", 0),
        "chunks_after_pruning": getattr(results, "chunks_after_pruning", 0),
        "pruning_reduction_pct": getattr(results, "pruning_reduction_pct", 0.0),
        "used_hybrid_search": getattr(results, "used_hybrid_search", False),
    }


async def recover_low_confidence(turn: Turn) -> None:
    """Self-correction retry, then deterministic absence proof, when the gate fails."""
    gate = turn.gate_result
    if not gate or gate.passed:
        return
    config = turn.config

    if getattr(config, "enable_self_correction", True) and turn.results:
        from app.services.self_correction_loop import SelfCorrectionPolicy
        tracer = _DictTracer(turn.trace)
        res = await SelfCorrectionPolicy(turn.db, turn.utility_llm, turn.engine._single_hybrid_search).execute_correction_loop(
            query=turn.retrieval_query or turn.req.content, project_id=turn.project_id, rag_config=config,
            old_confidence=gate.max_confidence, history_dicts=turn.history_dicts, tracer=tracer,
        )
        if res:
            chunks, new_gate = res
            turn.results = _to_results(chunks)
            turn.gate_result = new_gate
            turn.search_stats.update(chunks_after_pruning=len(chunks))
            if new_gate.passed:
                return

    from app.services.absence_prover import AbsenceProver
    proof = await run_in_threadpool(
        AbsenceProver(turn.db).prove_or_retry,
        query=turn.retrieval_query or turn.req.content, project_id=turn.project_id,
        hybrid_search_fn=turn.engine._single_hybrid_search, top_k=config.top_k,
    )
    if proof["action"] == "proven_absent":
        turn.not_found_proof = "verified_absent"
    elif proof["action"] == "retry_triggered" and proof["retry_chunks"]:
        from app.services.reranker_service import reranker_service
        retry, scored_flag = await run_in_threadpool(reranker_service.rerank_scored, turn.retrieval_query or turn.req.content, proof["retry_chunks"], max(config.top_k, 3))
        relevance = await run_in_threadpool(turn.engine.relevance_scores, turn.retrieval_query or turn.req.content, retry, config, scored_flag)
        retry = list(zip([d for d, _ in retry], relevance))
        retry_gate = turn.engine.evaluate_confidence(retry, config)
        turn.trace.setdefault("stages", {})["absence_prover_retry"] = {
            "stage": "absence_prover_retry", "status": "success" if retry_gate.passed else "failed",
            "metadata": {"retry_chunks_count": len(retry), "max_confidence": retry_gate.max_confidence},
        }
        if retry_gate.passed:
            turn.results = retry
            turn.gate_result = retry_gate


class _DictTracer:
    """Adapter so the self-correction loop can record stages into the trace dict."""

    def __init__(self, trace: dict):
        self.trace = trace

    def start_stage(self, stage):
        self.trace.setdefault("stages", {})[stage.value] = {"stage": stage.value, "start_time": time.time(), "status": "running", "metadata": {}}

    def end_stage(self, stage, status="success", metadata=None, error=None):
        stg = self.trace.get("stages", {}).get(stage.value)
        if stg:
            stg["end_time"] = time.time()
            stg["duration_ms"] = (stg["end_time"] - stg["start_time"]) * 1000.0
            stg["status"] = status
            stg["metadata"].update(metadata or {})
            if error:
                stg["error_message"] = error


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def sources_of(results: List[Tuple[LCDocument, float]]) -> List[dict]:
    out = []
    for i, (d, s) in enumerate(results, start=1):
        meta = d.metadata or {}
        out.append({
            "ref": i,
            "source": str(meta.get("source", "Unknown")),
            "doc_id": int(meta["doc_id"]) if meta.get("doc_id") not in (None, "") else 0,
            "chunk_id": meta.get("chunk_id"),
            "score": round(float(s), 4),
            "page": meta.get("page") or meta.get("page_no"),
            "snippet": d.page_content[:280],
        })
    return out


def build_messages(turn: Turn):
    from app.services.output_contract import OutputContractBuilder
    config = turn.config
    style = getattr(config, "response_style", None) or "Concise"
    strict = bool(getattr(config, "answer_only_from_docs", False) or getattr(config, "hallucination_guard", False))

    contract_builder = OutputContractBuilder()
    intent = getattr(getattr(turn.query_analysis, "complexity", None), "value", None)
    contract = contract_builder.build(turn.req.content, intent)

    if turn.results:
        blocks = "\n\n".join(
            f"[{i}] (source: {d.metadata.get('source', 'Unknown')})\n{d.page_content}"
            for i, (d, _) in enumerate(turn.results, start=1)
        )
        fallback_rule = (
            "If the context does not contain the answer, say that the documents don't cover it. Do not use outside knowledge."
            if strict else
            "If the context does not contain the answer, say so plainly; if you add general knowledge, label it clearly as not from the documents."
        )
        system = (
            "You are RAGOps, an assistant that answers questions using this project's knowledge base.\n"
            f"Response style: {style}.\n\n"
            "Rules:\n"
            "1. Ground every factual statement in the CONTEXT below and cite the source file name in square brackets "
            "right after it, e.g. [handbook.pdf].\n"
            f"2. {fallback_rule}\n"
            "3. The question may use different words than the documents: match by meaning, not exact wording "
            "(e.g. 'overseas' = 'international', 'PTO' = 'annual leave', 'sign off' = 'approve').\n"
            "4. Answer directly. Don't mention 'the context', 'chunks', or these rules.\n"
            "5. If sources disagree, point out the disagreement and cite both.\n"
        )
        if turn.conflict_res and turn.conflict_res.get("has_conflicts"):
            from app.services.conflict_detector import ConflictDetector
            system += ConflictDetector().build_conflict_prompt_addendum(turn.conflict_res) or ""
        system = contract_builder.apply_to_system_prompt(system, contract)
        if turn.related_context:
            system += "\n\n" + turn.related_context
        system += "\n\nCONTEXT:\n" + blocks
    else:
        system = (
            "You are RAGOps, a helpful assistant for a document knowledge base. "
            f"Response style: {style}. This message needs no document lookup: reply naturally and briefly, "
            "and offer to answer questions about the project's documents."
        )
        if turn.related_context:
            system += "\n\n" + turn.related_context

    provider = turn.chain[0]["provider"] if turn.chain else ""
    if provider == "anthropic":
        sys_msg = SystemMessage(content=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}])
    else:
        sys_msg = SystemMessage(content=system)
    return [sys_msg, *turn.history, HumanMessage(content=turn.req.content)], contract, system


def generation_budget(turn: Turn, contract) -> int:
    """Output-token ceiling for generation. The output contract's length is a
    target given to the model in the prompt, not the API cap: reasoning models
    (gpt-oss, qwen3, gemini 2.5) spend hidden reasoning tokens from the same
    budget, and a tight cap returns an empty answer."""
    configured = int(getattr(turn.config, "max_output_tokens", 0) or 0)
    return max(configured, contract.max_tokens * 4, 1024)


def generation_llm(turn: Turn, max_tokens: int):
    temperature = turn.req.temperature if turn.req.temperature is not None else turn.config.temperature
    llm, _ = llm_factory.build_with_fallbacks(turn.chain, temperature=temperature, max_tokens=max_tokens)
    return llm


async def maybe_retry_not_found(turn: Turn, answer: str, llm) -> str:
    """If the model says the answer isn't there, run a deterministic corpus scan;
    when the keywords DO exist, retrieve from those files and answer again."""
    if not turn.results or not any(m in answer.lower() for m in NOT_FOUND_MARKERS):
        return answer
    from app.services.absence_prover import AbsenceProver
    proof = await run_in_threadpool(
        AbsenceProver(turn.db).prove_or_retry,
        query=turn.retrieval_query or turn.req.content, project_id=turn.project_id,
        hybrid_search_fn=turn.engine._single_hybrid_search, top_k=turn.config.top_k,
    )
    if proof["action"] == "proven_absent":
        turn.not_found_proof = "verified_absent"
        return answer
    if proof["action"] != "retry_triggered" or not proof["retry_chunks"]:
        return answer
    existing = {d.metadata.get("chunk_id") for d, _ in turn.results}
    extra = [(d, s) for d, s in proof["retry_chunks"] if d.metadata.get("chunk_id") not in existing]
    if not extra:
        return answer
    turn.results = turn.results + _to_results(extra)
    messages, _, _ = build_messages(turn)
    retry = await llm.ainvoke(messages)
    turn.trace.setdefault("stages", {})["not_found_retry"] = {
        "stage": "not_found_retry", "status": "success", "metadata": {"extra_chunks": len(extra)},
    }
    return extract_text(retry.content)


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------

def evaluate_and_persist(
    turn: Turn, answer: str, contract, system_prompt: str, llm_status: str, llm_error: Optional[str],
    llm_ms: float, ttft_ms: Optional[float], usage: Optional[Dict[str, int]], model_used: str,
) -> dict:
    from app.services.output_contract import OutputVerifier
    from app.services.ragas_metrics import RAGASEvaluator

    db, config, content = turn.db, turn.config, turn.req.content
    ctx_chunks = [d.page_content for d, _ in turn.results]
    sources = sources_of(turn.results)
    latency_ms = (time.time() - turn.t0) * 1000.0

    stages = turn.trace.setdefault("stages", {})
    stages["llm_generation"] = {
        "stage": "llm_generation", "status": llm_status, "duration_ms": llm_ms,
        "metadata": {"model": model_used, "ttft_ms": ttft_ms}, "error_message": llm_error,
    }
    if llm_status == "failed":
        turn.trace["status"] = "failed"
    turn.trace["total_duration_ms"] = latency_ms

    sync_scores = evaluate_rag_response(content, answer, ctx_chunks)
    chunk_dicts = [{"content": t, "source": s["source"]} for t, s in zip(ctx_chunks, sources)]
    ragas_evaluator = RAGASEvaluator()
    ragas = ragas_evaluator.evaluate(query=content, retrieved_chunks=chunk_dicts, generated_answer=answer, existing_eval_scores=sync_scores)
    ragas_db = {
        "context_relevance": ragas.context_relevance, "faithfulness": ragas.faithfulness,
        "answer_relevance": ragas.answer_relevance, "groundedness": ragas.groundedness,
        "overall_score": ragas.overall_score, "hallucination_risk": ragas.hallucination_risk,
    }
    ragas_display = ragas_evaluator.to_display_dict(ragas)

    contract_res = OutputVerifier().verify(answer, contract, chunk_dicts)
    contract_res["format"] = contract.format.value

    claim_dict = None
    if getattr(config, "enable_claim_verification", True) and ctx_chunks:
        from app.services.claim_verifier import ClaimVerifier
        claims = ClaimVerifier().verify(answer, ctx_chunks)
        claim_dict = claims.to_dict()
        contract_res["unverified_claims"] = claims.unverified_claims if claims.has_unverified else []
        contract_res["claim_verification_score"] = claims.verification_score

    turn.trace["ragas_metrics"] = ragas_display["ragas_metrics"]
    turn.trace["output_contract"] = contract_res
    if claim_dict:
        turn.trace["claim_verification"] = claim_dict

    estimated = usage is None
    if usage is None:
        usage = {"input_tokens": int((len(system_prompt) + len(content)) / 4), "output_tokens": int(len(answer) / 4)}
    provider = turn.chain[0]["provider"] if turn.chain else "unknown"
    cost = calculate_cost(model_used, provider, usage["input_tokens"], usage["output_tokens"])
    tpot_ms = (
        round((llm_ms - ttft_ms) / max(usage["output_tokens"], 1), 2)
        if ttft_ms is not None and usage["output_tokens"] > 1 else None
    )
    quality = {
        "hallucination_score": float(sync_scores["hallucination_score"]),
        "faithfulness_score": float(sync_scores["faithfulness_score"]),
        "overall_quality_score": float(sync_scores["overall_quality_score"]),
        "quality_label": str(sync_scores["quality_label"]),
    }
    usage_metadata = {
        "model": model_used,
        "provider": provider,
        "temperature": turn.req.temperature,
        "history_limit": turn.req.history_limit,
        "rag_config": {
            "chunk_size": config.chunk_size, "chunk_overlap": config.chunk_overlap,
            "similarity_threshold": config.similarity_threshold, "top_k": config.top_k,
            "max_tokens": contract.max_tokens, "response_style": config.response_style,
        },
        "embeddings": config.embedding_model,
        "context_used": turn.related_titles or "None",
        "timestamp": datetime.utcnow().isoformat(),
        "tokens": {**usage, "estimated": estimated},
        "ttft_ms": ttft_ms,
        "quality": quality,
    }

    msg = Message(session_id=turn.chat_session.id, role="assistant", content=answer,
                  sources=json.dumps(sources), usage_metadata=usage_metadata, ragas_scores=ragas_db)
    db.add(msg)
    db.add(TokenUsage(project_id=turn.project_id, user_id=turn.user.id, session_id=turn.chat_session.id,
                      model=model_used, provider=provider, input_tokens=usage["input_tokens"],
                      output_tokens=usage["output_tokens"], total_tokens=usage["input_tokens"] + usage["output_tokens"], cost=cost))
    qlog = QueryLog(
        project_id=turn.project_id, user_id=turn.user.id, session_id=turn.chat_session.id,
        query_text=content, response_text=answer, model_used=f"{provider}/{model_used}",
        latency_ms=latency_ms, ttft_ms=ttft_ms, tpot_ms=tpot_ms, adaptive_backend_used=turn.adaptive_backend,
        chunks_retrieved=len(ctx_chunks), tokens_used=usage["input_tokens"] + usage["output_tokens"],
        citations_shown=len(sources), citations_clicked=0, context_chunks_json=json.dumps(ctx_chunks[:50]),
        hallucination_score=quality["hallucination_score"], faithfulness_score=quality["faithfulness_score"],
        chunks_before_pruning=turn.search_stats.get("chunks_before_pruning", 0),
        chunks_after_pruning=turn.search_stats.get("chunks_after_pruning", 0),
        pruning_reduction_pct=turn.search_stats.get("pruning_reduction_pct", 0.0),
        used_hybrid_search=turn.search_stats.get("used_hybrid_search", False),
        pipeline_trace=_json_safe(turn.trace), ragas_scores=ragas_db,
    )
    db.add(qlog)
    db.commit()
    db.refresh(qlog)
    get_cost_manager().breaker.record_spend(cost, turn.tier)

    payload = {
        "session_id": turn.chat_session.id,
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "usage_metadata": usage_metadata,
        "query_log_id": qlog.id,
        "quality": quality,
        "conflict_detection": turn.conflict_res,
        "not_found_proof": turn.not_found_proof,
        "ragas_metrics": ragas_display["ragas_metrics"],
        "output_contract": contract_res,
        "claim_verification": claim_dict,
        "pipeline_trace": _json_safe(turn.trace),
        "cached": False,
    }

    if (llm_status == "success" and answer.strip() and turn.cacheable and turn.results and turn.gate_result is not None
            and turn.gate_result.passed and getattr(config, "enable_semantic_cache", True)):
        try:
            # Keyed by the REQUESTED model, so switching models never returns another model's answer.
            semantic_cache.store(db, turn.project_id, turn.kb_version, content, turn.chain[0]["name"],
                                 {k: payload[k] for k in ("content", "sources", "quality", "ragas_metrics", "output_contract", "claim_verification", "conflict_detection")})
        except Exception as exc:
            logger.warning("Semantic cache store failed: %s", exc)
            db.rollback()
    return payload


def _json_safe(obj):
    return json.loads(json.dumps(obj, default=str))


def cached_response(turn: Turn) -> Optional[dict]:
    if not getattr(turn.config, "enable_semantic_cache", True):
        return None
    from app.services.query_rewriter import QueryRewriter
    if turn.history_dicts and QueryRewriter._needs_rewriting(None, turn.req.content):
        return None
    try:
        entry = semantic_cache.lookup(turn.db, turn.project_id, turn.kb_version, turn.req.content,
                                      float(getattr(turn.config, "semantic_cache_threshold", 0.92) or 0.92),
                                      model=turn.chain[0]["name"] if turn.chain else None)
    except Exception as exc:
        logger.warning("Semantic cache lookup failed: %s", exc)
        turn.db.rollback()
        return None
    if entry is None:
        return None
    data = dict(entry.response)
    latency_ms = (time.time() - turn.t0) * 1000.0
    trace = {"query": turn.req.content, "status": "success", "total_duration_ms": latency_ms,
             "stages": {"semantic_cache": {"stage": "semantic_cache", "status": "success", "duration_ms": latency_ms,
                                           "metadata": {"cache_hit": True, "matched_query": entry.query, "hits": entry.hits}}}}
    usage_metadata = {"model": entry.model, "provider": "cache", "context_used": "Semantic cache hit",
                      "timestamp": datetime.utcnow().isoformat(), "quality": data.get("quality"),
                      "cost_control": {"source": "cache"}}
    db = turn.db
    db.add(Message(session_id=turn.chat_session.id, role="assistant", content=data["content"],
                   sources=json.dumps(data.get("sources") or []), usage_metadata=usage_metadata))
    quality = data.get("quality") or {}
    qlog = QueryLog(project_id=turn.project_id, user_id=turn.user.id, session_id=turn.chat_session.id,
                    query_text=turn.req.content, response_text=data["content"], model_used=f"cache/{entry.model}",
                    latency_ms=latency_ms, chunks_retrieved=len(data.get("sources") or []), tokens_used=0,
                    citations_shown=len(data.get("sources") or []), hallucination_score=quality.get("hallucination_score"),
                    faithfulness_score=quality.get("faithfulness_score"), pipeline_trace=trace, adaptive_backend_used="cache")
    db.add(qlog)
    db.commit()
    db.refresh(qlog)
    return {**data, "session_id": turn.chat_session.id, "role": "assistant", "usage_metadata": usage_metadata,
            "query_log_id": qlog.id, "not_found_proof": None, "pipeline_trace": trace, "cached": True}


def refusal_response(turn: Turn) -> dict:
    answer = turn.gate_result.refusal_reason or "I couldn't find reliable information about this in the project's documents."
    if turn.not_found_proof == "verified_absent":
        answer = ("I couldn't find information about this in the project's documents. "
                  "None of them contain the key terms of your question, so try rephrasing "
                  "it with the wording your documents use.")
    latency_ms = (time.time() - turn.t0) * 1000.0
    stages = turn.trace.setdefault("stages", {})
    stages["llm_generation"] = {"stage": "llm_generation", "status": "skipped", "duration_ms": 0.0,
                                "metadata": {"reason": "confidence_gate_failed"}}
    turn.trace["status"] = "refused"
    turn.trace["total_duration_ms"] = latency_ms
    usage_metadata = {"model": None, "provider": None, "context_used": "Confidence gate refused",
                      "timestamp": datetime.utcnow().isoformat(),
                      "quality": {"hallucination_score": 0.0, "faithfulness_score": None, "overall_quality_score": None,
                                  "quality_label": "Refused (low confidence)"},
                      "confidence": {"max": turn.gate_result.max_confidence,
                                     "threshold": getattr(turn.config, "confidence_threshold", 0.4)}}
    db = turn.db
    db.add(Message(session_id=turn.chat_session.id, role="assistant", content=answer, sources="[]", usage_metadata=usage_metadata))
    qlog = QueryLog(project_id=turn.project_id, user_id=turn.user.id, session_id=turn.chat_session.id,
                    query_text=turn.req.content, response_text=answer, model_used="refused",
                    latency_ms=latency_ms, chunks_retrieved=len(turn.results), tokens_used=0,
                    chunks_before_pruning=turn.search_stats.get("chunks_before_pruning", 0),
                    chunks_after_pruning=turn.search_stats.get("chunks_after_pruning", 0),
                    used_hybrid_search=turn.search_stats.get("used_hybrid_search", False),
                    pipeline_trace=_json_safe(turn.trace), adaptive_backend_used=turn.adaptive_backend)
    db.add(qlog)
    db.commit()
    db.refresh(qlog)
    return {"session_id": turn.chat_session.id, "role": "assistant", "content": answer, "sources": [],
            "usage_metadata": usage_metadata, "query_log_id": qlog.id, "quality": usage_metadata["quality"],
            "conflict_detection": None, "not_found_proof": turn.not_found_proof, "ragas_metrics": None,
            "output_contract": None, "claim_verification": None, "pipeline_trace": _json_safe(turn.trace),
            "refused": True, "cached": False}


def schedule_title(turn: Turn, answer: str) -> None:
    if not turn.is_new_session or turn.req.title:
        return

    async def _title(session_id: int, chain: List[dict], content: str, answer: str):
        try:
            llm, _ = llm_factory.build_with_fallbacks(chain, temperature=0.2, max_tokens=1024, fast=True)
            res = await llm.ainvoke(
                "Summarize this conversation as a 3-5 word title. No quotes or punctuation at the end.\n"
                f"User: {content[:500]}\nAI: {answer[:500]}"
            )
            title = extract_text(res.content).replace('"', "").replace("'", "").replace("**", "").strip()[:50]
            if title:
                with Session(engine) as s:
                    cs = s.get(ChatSession, session_id)
                    if cs:
                        cs.title = title
                        s.add(cs)
                        s.commit()
        except Exception as exc:
            logger.info("Auto-title failed: %s", exc)

    task = asyncio.create_task(_title(turn.chat_session.id, turn.chain, turn.req.content, answer))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def _prepare_and_retrieve(req, db: Session, user: User) -> Tuple[Turn, Optional[dict]]:
    turn = await run_in_threadpool(prepare, req, db, user)
    turn.lap("prepare")
    hit = await run_in_threadpool(cached_response, turn)
    turn.lap("cache_lookup")
    if hit:
        return turn, hit
    allowed, reason = get_cost_manager().breaker.should_allow()
    if not allowed:
        raise HTTPException(status_code=503, detail="Spending limit reached for now. Please try again later.")
    await retrieve(turn)
    turn.lap("retrieve")
    await recover_low_confidence(turn)
    turn.lap("recover")
    if turn.gate_result is not None and not turn.gate_result.passed:
        turn.trace["timings_ms"] = turn.timings
        return turn, await run_in_threadpool(refusal_response, turn)
    return turn, None


async def run_turn(req, db: Session, user: User) -> dict:
    turn, early = await _prepare_and_retrieve(req, db, user)
    if early:
        early.setdefault("pipeline_trace", {})["timings_ms"] = turn.timings
        return early
    messages, contract, system_prompt = build_messages(turn)
    llm = generation_llm(turn, generation_budget(turn, contract))
    start = time.time()
    usage, model_used, status, error = None, turn.chain[0]["name"], "success", None
    try:
        result = await llm.ainvoke(messages)
        answer = extract_text(result.content)
        if not answer.strip():
            # Reasoning model exhausted its budget: retry once on the next model in the chain.
            logger.warning("Empty answer from %s; retrying with next model", turn.chain[0]["name"])
            retry_llm, _ = llm_factory.build_with_fallbacks(turn.chain[1:] or turn.chain, temperature=0.2, max_tokens=4096)
            result = await retry_llm.ainvoke(messages)
            answer = extract_text(result.content)
        usage = llm_factory.token_usage(result)
        model_used = llm_factory.response_model_name(result, model_used)
        answer = await maybe_retry_not_found(turn, answer, llm)
        if getattr(turn.config, "enable_reflective_generation", False) and turn.results:
            from app.services.reflective_generation import ReflectiveGenerator
            answer, _, stage, outcome = await ReflectiveGenerator(llm).generate_with_reflection(
                answer, turn.req.content, [d.page_content for d, _ in turn.results])
            turn.trace.setdefault("stages", {})["reflective_generation"] = {
                "stage": "reflective_generation", "status": "success", "metadata": {"stage_reached": stage, "outcome": outcome}}
    except Exception as exc:
        logger.exception("Generation failed")
        answer = "Sorry, every configured model failed to respond. Please try again in a moment."
        status, error = "failed", str(exc)[:500]
    llm_ms = (time.time() - start) * 1000.0
    turn.lap("generate")
    payload = await run_in_threadpool(evaluate_and_persist, turn, answer, contract, system_prompt, status, error, llm_ms, None, usage, model_used)
    turn.lap("evaluate_persist")
    payload["pipeline_trace"]["timings_ms"] = turn.timings
    schedule_title(turn, answer)
    return payload


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def stream_turn(req, db: Session, user: User) -> AsyncIterator[str]:
    """Server-Sent Events: `meta` (sources, retrieval trace), `token` deltas,
    then `done` with the same payload as the non-streaming endpoint."""
    try:
        turn, early = await _prepare_and_retrieve(req, db, user)
    except HTTPException as exc:
        yield _sse("error", {"status": exc.status_code, "detail": exc.detail})
        return
    if early:
        early.setdefault("pipeline_trace", {})["timings_ms"] = turn.timings
        yield _sse("token", {"delta": early["content"]})
        yield _sse("done", early)
        return

    yield _sse("meta", {"session_id": turn.chat_session.id, "sources": sources_of(turn.results),
                        "turn_type": turn.turn_type, "model_chain": turn.chain[:4]})
    messages, contract, system_prompt = build_messages(turn)
    llm = generation_llm(turn, generation_budget(turn, contract))
    start = time.time()
    ttft_ms, parts, usage = None, [], None
    model_used, status, error = turn.chain[0]["name"], "success", None
    try:
        async for chunk in llm.astream(messages):
            delta = extract_delta(chunk.content)
            if delta:
                if ttft_ms is None:
                    ttft_ms = round((time.time() - start) * 1000.0, 2)
                parts.append(delta)
                yield _sse("token", {"delta": delta})
            usage = llm_factory.token_usage(chunk) or usage
            model_used = llm_factory.response_model_name(chunk, model_used)
        answer = "".join(parts).strip()
        if not answer:
            raise RuntimeError("Model returned an empty response")
    except Exception as exc:
        logger.exception("Streaming generation failed")
        if parts:
            answer = "".join(parts).strip()
        else:
            answer = "Sorry, every configured model failed to respond. Please try again in a moment."
            yield _sse("token", {"delta": answer})
        status, error = "failed", str(exc)[:500]
    llm_ms = (time.time() - start) * 1000.0
    turn.lap("generate")
    payload = await run_in_threadpool(evaluate_and_persist, turn, answer, contract, system_prompt, status, error, llm_ms, ttft_ms, usage, model_used)
    turn.lap("evaluate_persist")
    payload["pipeline_trace"]["timings_ms"] = turn.timings
    schedule_title(turn, answer)
    yield _sse("done", payload)
