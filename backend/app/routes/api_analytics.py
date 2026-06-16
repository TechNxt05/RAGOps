from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from app.auth.deps import get_current_user, get_current_admin
from app.db import get_session
from app.models.query_log import QueryLog
from app.models.user import User, UserRole

router = APIRouter(prefix="/api/analytics", tags=["api-analytics"])


class CitationClickRequest(BaseModel):
    query_log_id: int
    citation_index: int = 0


@router.post("/citation-click")
def track_citation_click(
    req: CitationClickRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    log = session.get(QueryLog, req.query_log_id)
    if not log:
        return {"ok": True}
    if current_user.role != UserRole.ADMIN and log.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed")
    log.citations_clicked += 1
    session.add(log)
    session.commit()
    return {"ok": True}


@router.get("/{project_id}")
def get_project_analytics(
    project_id: int,
    days: int = 30,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_admin),
) -> dict[str, Any]:
    since = datetime.utcnow() - timedelta(days=days)
    logs = session.exec(
        select(QueryLog).where(QueryLog.project_id == project_id).where(QueryLog.created_at >= since)
    ).all()

    if not logs:
        return {
            "total_queries": 0,
            "avg_latency_ms": 0.0,
            "avg_hallucination_score": 0.0,
            "avg_faithfulness_score": 0.0,
            "citation_engagement_rate": 0.0,
            "avg_chunks_before_pruning": 0.0,
            "avg_chunks_after_pruning": 0.0,
            "avg_pruning_reduction_pct": 0.0,
            "hybrid_search_usage_pct": 0.0,
            "daily_volume": [],
            "model_breakdown": [],
            "quality_daily": [],
            "agentic_metrics": {
                "total_agentic_queries": 0,
                "agentic_success_rate": 0.0,
                "avg_agentic_attempts": 0.0,
                "most_common_fallbacks": []
            },
            "avg_context_relevance": 0.0,
            "avg_ragas_faithfulness": 0.0,
            "avg_answer_relevance": 0.0,
            "avg_groundedness": 0.0,
            "avg_overall_ragas": 0.0,
            "avg_compression_ratio": 0.0,
            "avg_cache_savings_usd": 0.0,
        }

    total = len(logs)
    latencies = [float(l.latency_ms or 0) for l in logs if l.latency_ms is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    hall_vals = [float(l.hallucination_score) for l in logs if l.hallucination_score is not None]
    faith_vals = [float(l.faithfulness_score) for l in logs if l.faithfulness_score is not None]
    avg_hall = sum(hall_vals) / len(hall_vals) if hall_vals else 0.0
    avg_faith = sum(faith_vals) / len(faith_vals) if faith_vals else 0.0

    cr_all = []
    f_all = []
    ar_all = []
    g_all = []
    o_all = []
    comp_ratios = []
    cache_savings = []

    for l in logs:
        if l.ragas_scores and isinstance(l.ragas_scores, dict):
            if "context_relevance" in l.ragas_scores:
                cr_all.append(float(l.ragas_scores["context_relevance"]))
            if "faithfulness" in l.ragas_scores:
                f_all.append(float(l.ragas_scores["faithfulness"]))
            if "answer_relevance" in l.ragas_scores:
                ar_all.append(float(l.ragas_scores["answer_relevance"]))
            if "groundedness" in l.ragas_scores:
                g_all.append(float(l.ragas_scores["groundedness"]))
            if "overall_score" in l.ragas_scores:
                o_all.append(float(l.ragas_scores["overall_score"]))
        if l.pipeline_trace and isinstance(l.pipeline_trace, dict):
            stats = l.pipeline_trace.get("compression_stats")
            if stats and isinstance(stats, dict):
                ratio = stats.get("compression_ratio")
                if ratio is not None:
                    comp_ratios.append(float(ratio))
            cache_info = l.pipeline_trace.get("prompt_cache_info")
            if cache_info and isinstance(cache_info, dict):
                usd = cache_info.get("estimated_monthly_savings_usd")
                if usd is not None:
                    cache_savings.append(float(usd))

    avg_context_relevance = sum(cr_all) / len(cr_all) if cr_all else 0.0
    avg_ragas_faithfulness = sum(f_all) / len(f_all) if f_all else 0.0
    avg_answer_relevance = sum(ar_all) / len(ar_all) if ar_all else 0.0
    avg_groundedness = sum(g_all) / len(g_all) if g_all else 0.0
    avg_overall_ragas = sum(o_all) / len(o_all) if o_all else 0.0
    avg_compression_ratio = sum(comp_ratios) / len(comp_ratios) if comp_ratios else 0.0
    avg_cache_savings = sum(cache_savings) / len(cache_savings) if cache_savings else 0.0

    cited = [l for l in logs if l.citations_shown > 0]
    engagement = (
        sum(l.citations_clicked for l in cited) / sum(l.citations_shown for l in cited)
        if cited and sum(l.citations_shown for l in cited) > 0
        else 0.0
    )

    by_date: dict[str, list[QueryLog]] = defaultdict(list)
    for row in logs:
        dkey = row.created_at.strftime("%Y-%m-%d")
        by_date[dkey].append(row)

    daily_volume = []
    for dkey in sorted(by_date.keys()):
        rows = by_date[dkey]
        lats = [float(r.latency_ms or 0) for r in rows if r.latency_ms is not None]
        daily_volume.append(
            {
                "date": dkey,
                "count": len(rows),
                "avg_latency": sum(lats) / len(lats) if lats else 0.0,
            }
        )

    model_counter: Counter[str] = Counter()
    model_latency: dict[str, list[float]] = defaultdict(list)
    for row in logs:
        model_counter[row.model_used] += 1
        if row.latency_ms is not None:
            model_latency[row.model_used].append(float(row.latency_ms))
    model_breakdown = [
        {
            "model": m,
            "count": c,
            "avg_latency": sum(model_latency[m]) / len(model_latency[m]) if model_latency[m] else 0.0,
        }
        for m, c in model_counter.items()
    ]

    quality_daily = []
    for dkey in sorted(by_date.keys()):
        rows = by_date[dkey]
        hv = [float(r.hallucination_score) for r in rows if r.hallucination_score is not None]
        fv = [float(r.faithfulness_score) for r in rows if r.faithfulness_score is not None]
        
        cr_vals = []
        f_vals = []
        ar_vals = []
        g_vals = []
        o_vals = []
        daily_comp_ratios = []
        for r in rows:
            if r.ragas_scores and isinstance(r.ragas_scores, dict):
                if "context_relevance" in r.ragas_scores:
                    cr_vals.append(float(r.ragas_scores["context_relevance"]))
                if "faithfulness" in r.ragas_scores:
                    f_vals.append(float(r.ragas_scores["faithfulness"]))
                if "answer_relevance" in r.ragas_scores:
                    ar_vals.append(float(r.ragas_scores["answer_relevance"]))
                if "groundedness" in r.ragas_scores:
                    g_vals.append(float(r.ragas_scores["groundedness"]))
                if "overall_score" in r.ragas_scores:
                    o_vals.append(float(r.ragas_scores["overall_score"]))
            if r.pipeline_trace and isinstance(r.pipeline_trace, dict):
                stats = r.pipeline_trace.get("compression_stats")
                if stats and isinstance(stats, dict):
                    ratio = stats.get("compression_ratio")
                    if ratio is not None:
                        daily_comp_ratios.append(float(ratio))
                        
        quality_daily.append(
            {
                "date": dkey,
                "avg_hallucination": sum(hv) / len(hv) if hv else None,
                "avg_faithfulness": sum(fv) / len(fv) if fv else None,
                "avg_context_relevance": sum(cr_vals) / len(cr_vals) if cr_vals else None,
                "avg_ragas_faithfulness": sum(f_vals) / len(f_vals) if f_vals else None,
                "avg_answer_relevance": sum(ar_vals) / len(ar_vals) if ar_vals else None,
                "avg_groundedness": sum(g_vals) / len(g_vals) if g_vals else None,
                "avg_overall_ragas": sum(o_vals) / len(o_vals) if o_vals else None,
                "avg_compression_ratio": sum(daily_comp_ratios) / len(daily_comp_ratios) if daily_comp_ratios else None
            }
        )

    pruned_before = [int(l.chunks_before_pruning or 0) for l in logs if l.chunks_before_pruning is not None]
    pruned_after = [int(l.chunks_after_pruning or 0) for l in logs if l.chunks_after_pruning is not None]
    pruning_reductions = [float(l.pruning_reduction_pct or 0.0) for l in logs if l.pruning_reduction_pct is not None]
    hybrid_searches = [1 if l.used_hybrid_search else 0 for l in logs]

    avg_before = sum(pruned_before) / len(pruned_before) if pruned_before else 0.0
    avg_after = sum(pruned_after) / len(pruned_after) if pruned_after else 0.0
    avg_reduction = sum(pruning_reductions) / len(pruning_reductions) if pruning_reductions else 0.0
    hybrid_usage_pct = (sum(hybrid_searches) / len(hybrid_searches) * 100.0) if hybrid_searches else 0.0

    # Agentic RAG Metrics
    agentic_logs = [
        l for l in logs 
        if l.pipeline_trace and isinstance(l.pipeline_trace, dict) and l.pipeline_trace.get("agentic")
    ]
    total_agentic = len(agentic_logs)
    agentic_success_rate = 0.0
    avg_agentic_attempts = 0.0
    most_common_fallbacks = []

    if total_agentic > 0:
        agentic_successes = sum(1 for l in agentic_logs if l.pipeline_trace.get("answered", True))
        agentic_success_rate = (agentic_successes / total_agentic) * 100.0
        avg_agentic_attempts = sum(l.pipeline_trace.get("attempts", 1) for l in agentic_logs) / total_agentic
        
        fallback_counter = Counter()
        for l in agentic_logs:
            strats = l.pipeline_trace.get("strategies_tried", [])
            if len(strats) > 1:
                for s in strats[1:]:
                    fallback_counter[s] += 1
        most_common_fallbacks = [
            {"strategy": k, "count": v} 
            for k, v in fallback_counter.items()
        ]

    return {
        "total_queries": total,
        "avg_latency_ms": round(avg_latency, 2),
        "avg_hallucination_score": round(avg_hall, 4),
        "avg_faithfulness_score": round(avg_faith, 4),
        "citation_engagement_rate": round(float(engagement), 4),
        "avg_chunks_before_pruning": round(avg_before, 2),
        "avg_chunks_after_pruning": round(avg_after, 2),
        "avg_pruning_reduction_pct": round(avg_reduction, 2),
        "hybrid_search_usage_pct": round(hybrid_usage_pct, 2),
        "daily_volume": daily_volume,
        "model_breakdown": model_breakdown,
        "quality_daily": quality_daily,
        "agentic_metrics": {
            "total_agentic_queries": total_agentic,
            "agentic_success_rate": round(agentic_success_rate, 2),
            "avg_agentic_attempts": round(avg_agentic_attempts, 2),
            "most_common_fallbacks": most_common_fallbacks
        },
        "avg_context_relevance": round(avg_context_relevance, 4),
        "avg_ragas_faithfulness": round(avg_ragas_faithfulness, 4),
        "avg_answer_relevance": round(avg_answer_relevance, 4),
        "avg_groundedness": round(avg_groundedness, 4),
        "avg_overall_ragas": round(avg_overall_ragas, 4),
        "avg_compression_ratio": round(avg_compression_ratio, 4),
        "avg_cache_savings_usd": round(avg_cache_savings, 2),
    }

