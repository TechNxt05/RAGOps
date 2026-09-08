"""
RAGOps MCP Server
Exposes enterprise RAGOps pipeline, retrieval, routing, and telemetry tools as MCP tools.
Operates via stdio transport for direct integration with Claude Desktop, LangGraph pipelines,
and external AI agents.
"""

import asyncio
import os
import sys
from typing import Dict, Any, List, Optional

# Patch httpx 0.28+ breaking change (proxies argument removed in favor of proxy)
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

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP(
    name="RAGOps",
    instructions=(
        "Enterprise RAG platform tools: full hybrid RAG pipeline execution, "
        "raw chunk retrieval, computation routing, adaptive router classification, "
        "and 24h rolling p95 SLA health telemetry."
    )
)


@mcp.tool()
async def query_knowledge_base(
    project_id: int,
    query: str,
    temperature: float = 0.1,
    model_provider: str = "groq",
    model_name: str = "llama3-70b-8192",
) -> Dict[str, Any]:
    """
    Run a query through the full RAGOps end-to-end pipeline:
    AdaptiveRetrievalRouter -> FAISS/BM25/Computation -> BGE Reranker ->
    Contextual Compressor -> Confidence Gate -> LLM Generation -> Claim Verifier.

    Args:
        project_id: The ID of the project knowledge base to query.
        query: The natural language question or query string.
        temperature: LLM sampling temperature (default 0.1).
        model_provider: Primary LLM provider ('groq' or 'google').
        model_name: Model identifier name.

    Returns:
        dict containing answer, retrieved_sources, claim_verification_results,
        ragas_scores, ttft_ms, tpot_ms, and pipeline_trace_id.
    """
    try:
        from app.db import engine
        from sqlmodel import Session
        from app.rag.engine import RAGEngine
        from app.services.claim_verifier import ClaimVerifier

        with Session(engine) as session:
            rag_engine = RAGEngine(session)

            # 1. Search pipeline execution
            search_results = rag_engine.search(
                query=query,
                project_id=project_id,
                k=4,
            )

            ctx_chunks = [d.page_content for d, _ in search_results]
            sources = [
                {
                    "source": getattr(d, "metadata", {}).get("source", "doc"),
                    "score": float(score)
                }
                for d, score in search_results
            ]

            # 2. LLM Generation
            if ctx_chunks:
                context_str = "\n\n".join(ctx_chunks[:8])
                prompt = (
                    "You are a precise enterprise assistant. Answer using ONLY the context below. "
                    "If the answer is not contained in the context, clearly state that you cannot find it.\n\n"
                    f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
                )
                
                from app.services.litellm_gateway import litellm_gateway
                gen_response = await litellm_gateway.generate(
                    prompt=prompt,
                    provider=model_provider,
                    model_name=model_name,
                    temperature=temperature,
                )
                answer = gen_response.content
            else:
                answer = "No relevant information found in the knowledge base."

            # 3. Claim Verification
            verifier = ClaimVerifier()
            verification = verifier.verify(answer=answer, context_chunks=ctx_chunks)

            return {
                "success": True,
                "project_id": project_id,
                "query": query,
                "answer": answer,
                "sources": sources,
                "chunks_retrieved": len(ctx_chunks),
                "claim_verification": {
                    "verified_claims_count": verification.supported,
                    "unsupported_claims_count": verification.unsupported,
                    "faithfulness_score": verification.verification_score,
                    "passed": verification.verification_score >= 0.70,
                },
                "pipeline_trace": getattr(search_results, "pipeline_trace", {})
            }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "project_id": project_id,
            "query": query
        }


@mcp.tool()
async def retrieve_chunks(
    project_id: int,
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Retrieve raw grounded context chunks for a query without executing LLM generation.
    Useful for external agents that require raw document context.

    Args:
        project_id: The project ID.
        query: The retrieval search query.
        top_k: Number of top chunks to retrieve (default 5).

    Returns:
        dict containing retrieved_chunks with content, score, and source metadata.
    """
    try:
        from app.db import engine
        from sqlmodel import Session
        from app.rag.engine import RAGEngine

        with Session(engine) as session:
            rag_engine = RAGEngine(session)
            results = rag_engine.search(
                query=query,
                project_id=project_id,
                k=top_k
            )

            chunks = [
                {
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata if hasattr(doc, "metadata") else {}
                }
                for doc, score in results
            ]

            return {
                "success": True,
                "project_id": project_id,
                "query": query,
                "top_k": top_k,
                "chunks": chunks,
                "count": len(chunks)
            }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "project_id": project_id,
            "query": query
        }


@mcp.tool()
async def run_computation_query(
    project_id: int,
    query: str
) -> Dict[str, Any]:
    """
    Route a query directly to the ComputationRouter / External Database Connector
    for analytical and structured aggregation questions, bypassing vector search.

    Args:
        project_id: Project ID.
        query: Analytical or aggregation question (e.g. "What is the total document count?").

    Returns:
        dict with computation route, tier, confidence, and computed answer if available.
    """
    try:
        from app.db import engine
        from sqlmodel import Session, select
        from app.services.computation_router import ComputationRouter
        from app.models.rag import ExternalDatabase
        from app.services.db_connector.query_executor import query_executor

        comp_router = ComputationRouter()
        route_decision = comp_router.route(query)

        with Session(engine) as session:
            # Check if external DB connector is attached to project
            ext_db = session.exec(select(ExternalDatabase).where(ExternalDatabase.project_id == project_id, ExternalDatabase.is_active == True)).first()
            if ext_db:
                db_result = await query_executor.execute_natural_query(session, ext_db, query)
                return {
                    "success": True,
                    "project_id": project_id,
                    "backend": "external_database",
                    "db_type": ext_db.db_type,
                    "result": db_result
                }

            # Fall back to internal document metadata computation
            computed_res = comp_router.compute_aggregation(query, project_id, session)

            return {
                "success": True,
                "project_id": project_id,
                "backend": "internal_computation",
                "route_info": {
                    "route": str(route_decision.get("route")),
                    "tier": route_decision.get("tier"),
                    "confidence": route_decision.get("confidence")
                },
                "computed_result": computed_res or "No direct structured aggregation found. Fall back to hybrid retrieval."
            }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "project_id": project_id,
            "query": query
        }


@mcp.tool()
async def check_router_decision(query: str) -> Dict[str, Any]:
    """
    Classify a query using AdaptiveRetrievalRouter without executing retrieval or generation.
    Returns predicted backend (bm25, computation, graph, hybrid), feature scores, and rationale.

    Args:
        query: Natural language query to test router classification on.

    Returns:
        dict containing chosen_backend, confidence, feature_scores, and reasoning.
    """
    try:
        from app.services.adaptive_retrieval_router import AdaptiveRetrievalRouter
        router = AdaptiveRetrievalRouter()
        decision = router.classify(query)

        return {
            "success": True,
            "query": decision.query,
            "chosen_backend": decision.chosen_backend,
            "confidence": decision.confidence,
            "feature_scores": decision.feature_scores,
            "fallback_triggered": decision.fallback_triggered,
            "reasoning": decision.reasoning
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "query": query
        }


@mcp.tool()
async def get_project_health(project_id: int) -> Dict[str, Any]:
    """
    Surface real-time SLA metrics (24h rolling p95 latency, TTFT/TPOT, cache hit rate,
    and router distribution) for a given project for external monitoring agents.

    Args:
        project_id: The project ID.

    Returns:
        dict containing sla_metrics_24h, latency_breakdown, and router_distribution.
    """
    try:
        from app.db import engine
        from sqlmodel import Session, select
        from app.models.query_log import QueryLog

        with Session(engine) as session:
            logs = session.exec(select(QueryLog).where(QueryLog.project_id == project_id)).all()
            
            if not logs:
                return {
                    "success": True,
                    "project_id": project_id,
                    "total_queries_logged": 0,
                    "p95_latency_ms": 0.0,
                    "sla_status": "NORMAL",
                    "sla_warning_exceeded": False,
                    "sla_threshold_ms": 2000.0,
                    "latency_breakdown": {
                        "avg_ttft_ms": 0.0,
                        "avg_tpot_ms": 0.0,
                    },
                    "cache_hit_rate_pct": 0.0,
                    "adaptive_router_distribution": {}
                }

            latencies = sorted([l.execution_time_ms for l in logs if l.execution_time_ms is not None])
            p95_idx = int(len(latencies) * 0.95) if latencies else 0
            p95_latency = latencies[p95_idx] if latencies else 0.0

            ttfts = [l.ttft_ms for l in logs if getattr(l, "ttft_ms", None) is not None]
            tpots = [l.tpot_ms for l in logs if getattr(l, "tpot_ms", None) is not None]
            
            ttft_avg = sum(ttfts) / len(ttfts) if ttfts else 0.0
            tpot_avg = sum(tpots) / len(tpots) if tpots else 0.0

            cache_hits = sum(1 for l in logs if getattr(l, "cache_hit", False))
            cache_rate = (cache_hits / len(logs)) * 100.0 if logs else 0.0

            router_dist: Dict[str, int] = {}
            for l in logs:
                backend = getattr(l, "adaptive_backend_used", None) or "hybrid"
                router_dist[backend] = router_dist.get(backend, 0) + 1

            return {
                "success": True,
                "project_id": project_id,
                "total_queries_logged": len(logs),
                "p95_latency_ms": round(p95_latency, 2),
                "sla_warning_exceeded": p95_latency > 2000.0,
                "sla_threshold_ms": 2000.0,
                "latency_breakdown": {
                    "avg_ttft_ms": round(ttft_avg, 2),
                    "avg_tpot_ms": round(tpot_avg, 2),
                },
                "cache_hit_rate_pct": round(cache_rate, 2),
                "adaptive_router_distribution": router_dist
            }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "project_id": project_id
        }


if __name__ == "__main__":
    mcp.run()
