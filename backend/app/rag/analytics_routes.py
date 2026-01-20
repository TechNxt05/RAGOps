from fastapi import APIRouter, Depends
from sqlmodel import Session, select, func
from app.db import get_session
from app.models.usage import TokenUsage
from app.models.user import User
from app.auth.deps import get_current_user
from typing import List, Dict, Any
from datetime import datetime, timedelta

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/summary")
def get_analytics_summary(
    project_id: int = None, 
    days: int = 30,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    # Base query
    query = select(TokenUsage).where(TokenUsage.user_id == current_user.id)
    if project_id:
        query = query.where(TokenUsage.project_id == project_id)
        
    start_date = datetime.utcnow() - timedelta(days=days)
    query = query.where(TokenUsage.timestamp >= start_date)
    
    logs = session.exec(query).all()
    
    total_requests = len(logs)
    total_cost = sum(l.cost for l in logs)
    total_tokens = sum(l.total_tokens for l in logs)
    
    # Group by Date
    daily_stats = {}
    for log in logs:
        date_str = log.timestamp.strftime("%Y-%m-%d")
        if date_str not in daily_stats:
            daily_stats[date_str] = {"requests": 0, "cost": 0, "tokens": 0}
        daily_stats[date_str]["requests"] += 1
        daily_stats[date_str]["cost"] += log.cost
        daily_stats[date_str]["tokens"] += log.total_tokens
        
    chart_data = [
        {"date": k, "requests": v["requests"], "cost": round(v["cost"], 4), "tokens": v["tokens"]}
        for k, v in daily_stats.items()
    ]
    chart_data.sort(key=lambda x: x["date"])
    
    # Model Distribution
    model_counts = {}
    for log in logs:
        if log.model not in model_counts:
            model_counts[log.model] = 0
        model_counts[log.model] += 1
        
    model_data = [{"name": k, "value": v} for k, v in model_counts.items()]
    
    return {
        "total_requests": total_requests,
        "total_cost": round(total_cost, 4),
        "total_tokens": total_tokens,
        "chart_data": chart_data,
        "model_distribution": model_data
    }
