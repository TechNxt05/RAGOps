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
            "daily_volume": [],
            "model_breakdown": [],
            "quality_daily": [],
        }

    total = len(logs)
    latencies = [float(l.latency_ms or 0) for l in logs if l.latency_ms is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    hall_vals = [float(l.hallucination_score) for l in logs if l.hallucination_score is not None]
    faith_vals = [float(l.faithfulness_score) for l in logs if l.faithfulness_score is not None]
    avg_hall = sum(hall_vals) / len(hall_vals) if hall_vals else 0.0
    avg_faith = sum(faith_vals) / len(faith_vals) if faith_vals else 0.0

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
        quality_daily.append(
            {
                "date": dkey,
                "avg_hallucination": sum(hv) / len(hv) if hv else None,
                "avg_faithfulness": sum(fv) / len(fv) if fv else None,
            }
        )

    return {
        "total_queries": total,
        "avg_latency_ms": round(avg_latency, 2),
        "avg_hallucination_score": round(avg_hall, 4),
        "avg_faithfulness_score": round(avg_faith, 4),
        "citation_engagement_rate": round(float(engagement), 4),
        "daily_volume": daily_volume,
        "model_breakdown": model_breakdown,
        "quality_daily": quality_daily,
    }
