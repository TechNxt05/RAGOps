"""
AutoRAGBench — Benchmark API Routes

Provides endpoints for:
- Querying pre-configured model profiles (1-click presets)
- Querying available RAG pipeline variants
- Starting and monitoring asynchronous benchmark evaluation runs
- Fetching live streaming progress, leaderboard results, and historical reports
"""

import os
import time
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.db import get_session
from app.auth.deps import get_current_admin, get_current_user
from app.models.user import User
from app.benchmark.variants import PipelineVariant, list_variants, get_variant_config
from app.benchmark.model_profiles import get_available_profiles, get_profile_by_id, MODEL_PROFILES
from app.benchmark.runner import BenchmarkRunner, BenchmarkQuery

logger = logging.getLogger("AutoRAGBenchAPI")

router = APIRouter(prefix="/api/benchmark", tags=["Benchmark"])

# In-memory benchmark job tracker
BENCHMARK_JOBS: Dict[str, Dict[str, Any]] = {}


class BenchmarkStartRequest(BaseModel):
    project_id: int = 1
    model_profile_id: str = "openrouter_gpt4o_mini"
    variants: List[str] = Field(default_factory=lambda: ["naive", "advanced", "hybrid", "full_ragops", "adaptive", "policy_driven"])
    dataset_type: str = "master_35"  # "smoke_5" | "master_35" | "custom"
    custom_queries: Optional[List[Dict[str, Any]]] = None
    delay_ms: int = 0
    resume: bool = False


@router.get("/models")
def get_model_profiles(_: User = Depends(get_current_user)):
    """Returns all pre-configured 1-click model profiles with provider key status."""
    return {"models": get_available_profiles()}


@router.get("/variants")
def get_pipeline_variants(_: User = Depends(get_current_user)):
    """Returns all 10 canonical RAG pipeline variants with metadata."""
    return {"variants": list_variants()}


@router.get("/datasets")
def get_benchmark_datasets(_: User = Depends(get_current_user)):
    """Returns available benchmark query suites."""
    return {
        "datasets": [
            {
                "id": "master_35",
                "name": "Master Ground-Truth Benchmark Suite (35 Queries)",
                "description": "Comprehensive evaluation covering 6 difficulty tiers (Factoid, Definitional, Analytical, Comparative, Multi-Hop, Absent).",
                "query_count": 35,
                "recommended": True
            },
            {
                "id": "smoke_5",
                "name": "Quick Verification Smoke Test (5 Queries)",
                "description": "Fast 5-query subset (1 Factoid, 1 Definitional, 1 Analytical, 1 Comparative, 1 Absent) for rapid sanity checks in under 1 minute.",
                "query_count": 5,
                "recommended": False
            }
        ]
    }


def _build_queries_for_dataset(dataset_type: str, custom_queries: Optional[List[Dict[str, Any]]] = None) -> List[BenchmarkQuery]:
    if dataset_type == "custom" and custom_queries:
        return [
            BenchmarkQuery(
                id=q.get("id", f"custom_{i}"),
                query=q["query"],
                category=q.get("category", "custom"),
                expected_sources=q.get("expected_sources"),
                expected_answer=q.get("expected_answer")
            )
            for i, q in enumerate(custom_queries)
        ]

    # Load master 35
    all_queries = []
    try:
        from scratch.labeled_queries import get_labeled_queries
        labeled = get_labeled_queries()
        for q in labeled:
            all_queries.append(BenchmarkQuery(
                id=q["id"],
                query=q["query"],
                category=q["category"],
                expected_sources=q.get("expected_sources")
            ))
    except Exception:
        from app.benchmark.runner import DEFAULT_BENCHMARK_QUERIES
        all_queries = DEFAULT_BENCHMARK_QUERIES

    if dataset_type == "smoke_5":
        # Pick 1 query per key category
        selected = []
        categories_seen = set()
        for q in all_queries:
            if q.category not in categories_seen and len(selected) < 5:
                selected.append(q)
                categories_seen.add(q.category)
        return selected if selected else all_queries[:5]

    return all_queries


async def _run_benchmark_job_async(job_id: str, req: BenchmarkStartRequest):
    """Background task executing the AutoRAGBench run."""
    job = BENCHMARK_JOBS.get(job_id)
    if not job:
        return

    profile = get_profile_by_id(req.model_profile_id) or MODEL_PROFILES.get("openrouter_gpt4o_mini")
    selected_variants = []
    for v_str in req.variants:
        try:
            selected_variants.append(PipelineVariant(v_str))
        except ValueError:
            pass

    if not selected_variants:
        selected_variants = [PipelineVariant.ADVANCED, PipelineVariant.FULL_RAGOPS]

    queries = _build_queries_for_dataset(req.dataset_type, req.custom_queries)
    total_steps = len(queries) * len(selected_variants)

    job["total_queries"] = total_steps
    job["status"] = "running"
    job["start_time"] = time.time()

    def progress_callback(event_data: Dict[str, Any]):
        if job.get("status") == "cancelled":
            raise asyncio.CancelledError("Benchmark cancelled by user.")
        job["completed_queries"] = event_data.get("completed_queries", 0)
        job["current_variant"] = event_data.get("variant_label", event_data.get("variant"))
        job["current_query"] = event_data.get("query_text", "")
        job["progress_pct"] = round((job["completed_queries"] / max(total_steps, 1)) * 100, 1)
        job["elapsed_seconds"] = round(time.time() - job["start_time"], 1)
        
        # Keep last 15 live completed queries
        live_list = job.get("live_events", [])
        live_list.append(event_data)
        if len(live_list) > 20:
            live_list.pop(0)
        job["live_events"] = live_list

    try:
        from app.db import engine
        from sqlmodel import Session
        
        output_file = f"benchmark_report_{job_id[:8]}.json"
        
        with Session(engine) as session:
            runner = BenchmarkRunner(
                project_id=req.project_id,
                session_db=session,
                user=None,
                queries=queries,
                variants=selected_variants,
                resume=req.resume,
                output_file=output_file,
                delay_ms=req.delay_ms,
                provider_override=profile.provider,
                model_override=profile.model_name,
                progress_callback=progress_callback,
            )
            
            summaries = await runner.run_all()
            
            # Serialize report
            json_report = runner.to_json(summaries)
            job["report_json"] = json_report
            job["summaries"] = [s.__dict__ for s in summaries]
            job["status"] = "completed"
            job["progress_pct"] = 100.0
            job["elapsed_seconds"] = round(time.time() - job["start_time"], 1)
            
            # Save master report file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_report)
            job["output_file"] = output_file
            
    except asyncio.CancelledError:
        job["status"] = "cancelled"
        job["error"] = "Benchmark was cancelled."
    except Exception as exc:
        logger.exception(f"AutoRAGBench job {job_id} failed: {exc}")
        job["status"] = "failed"
        job["error"] = str(exc)


@router.post("/start")
async def start_benchmark(
    req: BenchmarkStartRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_admin),
):
    """Starts an asynchronous AutoRAGBench evaluation run."""
    job_id = str(uuid.uuid4())
    profile = get_profile_by_id(req.model_profile_id)
    
    if not profile:
        raise HTTPException(status_code=400, detail=f"Invalid model profile: {req.model_profile_id}")

    BENCHMARK_JOBS[job_id] = {
        "job_id": job_id,
        "project_id": req.project_id,
        "model_id": profile.id,
        "model_name": profile.name,
        "model_provider": profile.provider,
        "model_api_name": profile.model_name,
        "variants": req.variants,
        "dataset_type": req.dataset_type,
        "status": "queued",
        "progress_pct": 0.0,
        "completed_queries": 0,
        "total_queries": 0,
        "current_variant": "",
        "current_query": "",
        "elapsed_seconds": 0.0,
        "live_events": [],
        "summaries": [],
        "error": None,
        "created_at": time.time(),
    }

    background_tasks.add_task(_run_benchmark_job_async, job_id, req)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"AutoRAGBench started with {profile.name} across {len(req.variants)} variants."
    }


@router.get("/status/{job_id}")
def get_benchmark_status(job_id: str, _: User = Depends(get_current_user)):
    """Poll live progress and recent activity for a running benchmark."""
    job = BENCHMARK_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Benchmark job not found.")

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "model_name": job["model_name"],
        "model_provider": job["model_provider"],
        "progress_pct": job["progress_pct"],
        "completed_queries": job["completed_queries"],
        "total_queries": job["total_queries"],
        "current_variant": job["current_variant"],
        "current_query": job["current_query"],
        "elapsed_seconds": job.get("elapsed_seconds", 0.0),
        "live_events": job.get("live_events", []),
        "error": job.get("error"),
    }


@router.get("/results/{job_id}")
def get_benchmark_results(job_id: str, _: User = Depends(get_current_user)):
    """Retrieve full evaluation metrics, leaderboard, and summaries for a completed job."""
    job = BENCHMARK_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Benchmark job not found.")

    if job["status"] != "completed" and not job.get("summaries"):
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress_pct": job["progress_pct"],
            "message": "Benchmark is still running. Please poll status."
        }

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "model_name": job["model_name"],
        "model_provider": job["model_provider"],
        "elapsed_seconds": job.get("elapsed_seconds", 0.0),
        "summaries": job.get("summaries", []),
        "output_file": job.get("output_file"),
    }


@router.post("/cancel/{job_id}")
def cancel_benchmark(job_id: str, current_user: User = Depends(get_current_admin)):
    """Cancels an active benchmark run."""
    job = BENCHMARK_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Benchmark job not found.")

    job["status"] = "cancelled"
    job["error"] = "Cancelled by user request."
    return {"status": "cancelled", "job_id": job_id}


@router.get("/history")
def get_benchmark_history(_: User = Depends(get_current_user)):
    """Lists past benchmark report files and cached results."""
    history = []
    # Check in-memory jobs
    for j_id, j_data in BENCHMARK_JOBS.items():
        if j_data.get("status") in ("completed", "failed"):
            history.append({
                "job_id": j_id,
                "model_name": j_data.get("model_name"),
                "status": j_data.get("status"),
                "elapsed_seconds": j_data.get("elapsed_seconds"),
                "variants_count": len(j_data.get("variants", [])),
                "created_at": j_data.get("created_at"),
            })
    return {"history": history}
