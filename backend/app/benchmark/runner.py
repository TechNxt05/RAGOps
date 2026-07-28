"""
AutoRAGBench — Benchmark Runner

Executes each pipeline variant against a dataset of benchmark queries,
collecting per-query metrics: RAGAS triad, groundedness, hallucination rate,
latency, token cost, and cache hit status.
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from app.benchmark.variants import PipelineVariant, get_variant_config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class BenchmarkQuery:
    """A single benchmark test case."""
    id: str
    query: str
    expected_answer: Optional[str] = None
    expected_sources: Optional[List[str]] = None
    category: str = "general"


@dataclass
class QueryResult:
    """Result of running one query through one variant."""
    query_id: str
    variant: str
    answer: str = ""
    latency_ms: float = 0.0
    token_cost_estimate: float = 0.0
    cache_hit: bool = False
    chunks_retrieved: int = 0
    ragas_context_relevance: float = 0.0
    ragas_faithfulness: float = 0.0
    ragas_answer_relevance: float = 0.0
    ragas_groundedness: float = 0.0
    ragas_overall: float = 0.0
    hallucination_risk: str = "unknown"
    confidence_score: float = 0.0
    error: Optional[str] = None


@dataclass
class VariantSummary:
    """Aggregate metrics for a single variant across all queries."""
    variant: str
    label: str
    description: str
    total_queries: int = 0
    avg_latency_ms: float = 0.0
    avg_context_relevance: float = 0.0
    avg_faithfulness: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_groundedness: float = 0.0
    avg_overall_score: float = 0.0
    total_token_cost: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    query_results: List[QueryResult] = field(default_factory=list)


# ------------------------------------------------------------------
# Default benchmark dataset
# ------------------------------------------------------------------

DEFAULT_BENCHMARK_QUERIES: List[BenchmarkQuery] = [
    BenchmarkQuery(
        id="factoid_01",
        query="What is the main purpose of this project?",
        category="factoid"
    ),
    BenchmarkQuery(
        id="factoid_02",
        query="Who is the author or creator of the documents?",
        category="factoid"
    ),
    BenchmarkQuery(
        id="analytical_01",
        query="Summarize the key findings across all uploaded documents.",
        category="analytical"
    ),
    BenchmarkQuery(
        id="comparative_01",
        query="Compare the different approaches mentioned in the documents.",
        category="comparative"
    ),
    BenchmarkQuery(
        id="procedural_01",
        query="What are the step-by-step instructions described?",
        category="procedural"
    ),
    BenchmarkQuery(
        id="absent_01",
        query="What does the document say about quantum computing in 2099?",
        category="absent"
    ),
    BenchmarkQuery(
        id="specific_01",
        query="List all specific numbers, dates, or statistics mentioned.",
        category="specific"
    ),
    BenchmarkQuery(
        id="follow_up_01",
        query="Can you elaborate on the first point you mentioned?",
        category="follow_up"
    ),
]


class BenchmarkRunner:
    """
    Orchestrates benchmark runs across all 6 pipeline variants.

    Usage:
        runner = BenchmarkRunner(project_id=1, session_db=db, user=admin_user)
        results = await runner.run_all()
        report = runner.generate_summary(results)
    """

    def __init__(
        self,
        project_id: int,
        session_db,
        user,
        queries: Optional[List[BenchmarkQuery]] = None,
        variants: Optional[List[PipelineVariant]] = None,
    ):
        self.project_id = project_id
        self.session_db = session_db
        self.user = user
        self.queries = queries or DEFAULT_BENCHMARK_QUERIES
        self.variants = variants or list(PipelineVariant)

    async def _run_single_query(
        self,
        query: BenchmarkQuery,
        variant: PipelineVariant,
        variant_config: Dict[str, Any],
    ) -> QueryResult:
        """
        Execute a single query against a specific pipeline variant.

        This constructs a synthetic RAGConfig override and runs through the
        RAG engine with only the stages enabled for this variant.
        """
        result = QueryResult(query_id=query.id, variant=variant.value)
        t0 = time.time()

        try:
            from sqlmodel import select
            from app.models.rag import RAGConfig
            from app.rag.engine import RAGEngine

            engine = RAGEngine(self.session_db)

            # Get active config as base and apply variant overrides
            try:
                rag_config = engine.get_active_config(self.project_id)
            except ValueError:
                rag_config = RAGConfig(project_id=self.project_id)

            # Override toggles
            rag_config.use_hybrid_search = variant_config.get("use_hybrid_search", True)
            rag_config.enable_self_correction = variant_config.get("use_self_correction", True)
            rag_config.enable_adaptive_router = variant_config.get("enable_adaptive_router", False)

            if rag_config.enable_adaptive_router:
                from app.services.adaptive_retrieval_router import AdaptiveRetrievalRouter
                adaptive_router = AdaptiveRetrievalRouter()
                decision = adaptive_router.classify(query.query)
                if decision.chosen_backend == "bm25":
                    rag_config.use_hybrid_search = False

            # Build LLM client
            import os as _os
            if rag_config.primary_llm_provider == "groq":
                from langchain_groq import ChatGroq
                llm_client = ChatGroq(
                    model_name=rag_config.primary_llm_name,
                    api_key=_os.getenv("GROQ_API_KEY"),
                    temperature=rag_config.temperature,
                )
            else:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm_client = ChatGoogleGenerativeAI(
                    model=rag_config.primary_llm_name,
                    google_api_key=_os.getenv("GEMINI_API_KEY"),
                    temperature=rag_config.temperature,
                )

            # --- Execute retrieval stage ---
            search_results = engine.search(
                query=query.query,
                project_id=self.project_id,
                k=rag_config.top_k,
                score_threshold=rag_config.similarity_threshold,
                llm_client=llm_client if variant_config.get("use_query_rewriting") else None,
            )

            ctx_chunks = [d.page_content for d, _ in search_results]
            result.chunks_retrieved = len(ctx_chunks)

            # Compute confidence from gate result
            gate_res = getattr(search_results, "confidence_gate_result", None)
            if gate_res:
                result.confidence_score = gate_res.max_confidence

            # --- Generate answer ---
            if ctx_chunks:
                context_str = "\n\n".join(ctx_chunks[:12])
                gen_prompt = (
                    "You are a precise assistant. Answer using ONLY the context below. "
                    "If the answer is not in the context, say you cannot find it.\n\n"
                    f"Context:\n{context_str}\n\nQuestion: {query.query}\nAnswer:"
                )
                gen_res = await llm_client.ainvoke(gen_prompt)
                result.answer = gen_res.content if hasattr(gen_res, "content") else str(gen_res)
            else:
                result.answer = "No relevant information found in the knowledge base."

            # --- Evaluate with RAGAS ---
            from app.services.ragas_metrics import RAGASEvaluator
            from app.services.rag_evaluator import evaluate_rag_response

            evaluator = RAGASEvaluator()
            sync_scores = evaluate_rag_response(query.query, result.answer, ctx_chunks)
            retrieved_chunks_dicts = [{"content": t, "source": ""} for t in ctx_chunks]

            ragas = evaluator.evaluate(
                query=query.query,
                retrieved_chunks=retrieved_chunks_dicts,
                generated_answer=result.answer,
                existing_eval_scores=sync_scores,
            )

            result.ragas_context_relevance = ragas.context_relevance
            result.ragas_faithfulness = ragas.faithfulness
            result.ragas_answer_relevance = ragas.answer_relevance
            result.ragas_groundedness = ragas.groundedness
            result.ragas_overall = ragas.overall_score
            result.hallucination_risk = ragas.hallucination_risk

            # Token cost estimate
            i_tokens = int(len(gen_prompt) / 4) if ctx_chunks else 0
            o_tokens = int(len(result.answer) / 4)
            from app.utils.cost import calculate_cost
            result.token_cost_estimate = calculate_cost(
                rag_config.primary_llm_name,
                rag_config.primary_llm_provider,
                i_tokens,
                o_tokens,
            )

            # Cache hit detection
            result.cache_hit = getattr(search_results, "pipeline_trace", {}).get("retrieval_source") == "cache"

        except Exception as exc:
            result.error = str(exc)
            try:
                self.session_db.rollback()
            except Exception:
                pass
            logger.error(f"Benchmark error [{variant.value}][{query.id}]: {exc}")

        result.latency_ms = (time.time() - t0) * 1000.0
        return result

    async def run_variant(self, variant: PipelineVariant) -> VariantSummary:
        """Run all benchmark queries against a single variant."""
        config = get_variant_config(variant)
        summary = VariantSummary(
            variant=variant.value,
            label=config["label"],
            description=config["description"],
        )

        for query in self.queries:
            result = await self._run_single_query(query, variant, config)
            summary.query_results.append(result)

        # Aggregate metrics
        n = len(summary.query_results)
        summary.total_queries = n
        if n > 0:
            ok_results = [r for r in summary.query_results if r.error is None]
            n_ok = len(ok_results) or 1
            summary.avg_latency_ms = sum(r.latency_ms for r in summary.query_results) / n
            summary.avg_context_relevance = sum(r.ragas_context_relevance for r in ok_results) / n_ok
            summary.avg_faithfulness = sum(r.ragas_faithfulness for r in ok_results) / n_ok
            summary.avg_answer_relevance = sum(r.ragas_answer_relevance for r in ok_results) / n_ok
            summary.avg_groundedness = sum(r.ragas_groundedness for r in ok_results) / n_ok
            summary.avg_overall_score = sum(r.ragas_overall for r in ok_results) / n_ok
            summary.total_token_cost = sum(r.token_cost_estimate for r in ok_results)
            summary.cache_hit_rate = sum(1 for r in ok_results if r.cache_hit) / n_ok
            summary.error_rate = sum(1 for r in summary.query_results if r.error is not None) / n

        return summary

    async def run_all(self) -> List[VariantSummary]:
        """Execute all variants sequentially and return summaries."""
        summaries = []
        for variant in self.variants:
            logger.info(f"[AutoRAGBench] Running variant: {variant.value}")
            summary = await self.run_variant(variant)
            summaries.append(summary)
            logger.info(
                f"[AutoRAGBench] {variant.value} complete — "
                f"avg_overall={summary.avg_overall_score:.3f}, "
                f"avg_latency={summary.avg_latency_ms:.0f}ms, "
                f"cost=${summary.total_token_cost:.4f}"
            )
        return summaries

    def to_json(self, summaries: List[VariantSummary]) -> str:
        """Serialize all summaries to JSON string."""
        return json.dumps([asdict(s) for s in summaries], indent=2, default=str)
