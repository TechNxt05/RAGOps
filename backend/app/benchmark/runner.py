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
    retrieved_chunk_ids: List[int] = field(default_factory=list)


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
    
    # Deterministic Retrieval Metrics
    avg_recall_at_5: float = 0.0
    avg_precision_at_5: float = 0.0
    avg_mrr: float = 0.0
    avg_ndcg_at_5: float = 0.0
    
    total_token_cost: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    query_results: List[QueryResult] = field(default_factory=list)


# ------------------------------------------------------------------
# Default benchmark dataset
# ------------------------------------------------------------------

DEFAULT_BENCHMARK_QUERIES: List[BenchmarkQuery] = [
    # FACTOID
    BenchmarkQuery(id="factoid_01", query="What is a qubit?", category="factoid", expected_sources=["1", "2"]),
    BenchmarkQuery(id="factoid_02", query="What is the difference between a qubit and a classical bit?", category="factoid", expected_sources=["3"]),
    BenchmarkQuery(id="factoid_03", query="Who proposed the idea of a quantum Turing machine?", category="factoid", expected_sources=["10"]),
    BenchmarkQuery(id="factoid_04", query="What year did Peter Shor develop Shor's algorithm?", category="factoid", expected_sources=["15"]),
    BenchmarkQuery(id="factoid_05", query="What is quantum decoherence?", category="factoid", expected_sources=["12", "13"]),
    BenchmarkQuery(id="factoid_06", query="What is the threshold theorem?", category="factoid", expected_sources=["18"]),
    BenchmarkQuery(id="factoid_07", query="What physical systems are used to build qubits?", category="factoid"),
    BenchmarkQuery(id="factoid_08", query="Name two types of quantum logic gates.", category="factoid"),
    BenchmarkQuery(id="factoid_09", query="What does NISQ stand for?", category="factoid"),
    BenchmarkQuery(id="factoid_10", query="What is quantum supremacy?", category="factoid"),
    
    # DEFINITIONAL
    BenchmarkQuery(id="definitional_01", query="Define quantum entanglement in the context of computing.", category="definitional"),
    BenchmarkQuery(id="definitional_02", query="What is quantum superposition?", category="definitional"),
    BenchmarkQuery(id="definitional_03", query="Define quantum error correction.", category="definitional"),
    BenchmarkQuery(id="definitional_04", query="What is adiabatic quantum computing?", category="definitional"),
    BenchmarkQuery(id="definitional_05", query="Explain the concept of quantum annealing.", category="definitional"),
    
    # ANALYTICAL
    BenchmarkQuery(id="analytical_01", query="How does Shor's algorithm threaten RSA encryption?", category="analytical"),
    BenchmarkQuery(id="analytical_02", query="Why is maintaining coherence difficult in a quantum computer?", category="analytical"),
    BenchmarkQuery(id="analytical_03", query="What are the main obstacles to building a large-scale quantum computer?", category="analytical"),
    BenchmarkQuery(id="analytical_04", query="How does Grover's algorithm improve database search?", category="analytical"),
    BenchmarkQuery(id="analytical_05", query="What implications does quantum computing have for materials science?", category="analytical"),

    # COMPARATIVE
    BenchmarkQuery(id="comparative_01", query="Compare superconducting qubits with trapped ion qubits.", category="comparative"),
    BenchmarkQuery(id="comparative_02", query="How does a quantum computer's memory differ from a classical computer's?", category="comparative"),
    BenchmarkQuery(id="comparative_03", query="Contrast quantum gate arrays with quantum annealing.", category="comparative"),
    BenchmarkQuery(id="comparative_04", query="What is the difference between a physical qubit and a logical qubit?", category="comparative"),
    BenchmarkQuery(id="comparative_05", query="How does Grover's speedup compare to Shor's speedup?", category="comparative"),

    # MULTI-HOP
    BenchmarkQuery(id="multihop_01", query="Based on the history, how did Feynman's original proposal connect to Shor's later discovery?", category="multi_hop"),
    BenchmarkQuery(id="multihop_02", query="How does the physical implementation of a qubit relate to the need for quantum error correction?", category="multi_hop"),
    BenchmarkQuery(id="multihop_03", query="If NISQ computers cannot perform fault-tolerant error correction, how do they attempt to achieve quantum advantage?", category="multi_hop"),
    BenchmarkQuery(id="multihop_04", query="How do decoherence rates affect the depth of the quantum circuits that can be executed?", category="multi_hop"),
    BenchmarkQuery(id="multihop_05", query="Trace the relationship between cryptography vulnerabilities and the development of post-quantum cryptography.", category="multi_hop"),

    # PROCEDURAL
    BenchmarkQuery(id="procedural_01", query="How do you perform a measurement on a qubit?", category="procedural"),
    BenchmarkQuery(id="procedural_02", query="What is the general process for running a quantum algorithm?", category="procedural"),
    BenchmarkQuery(id="procedural_03", query="How does the surface code implement error correction?", category="procedural"),
    
    # ABSENT (Hallucination checks)
    BenchmarkQuery(id="absent_01", query="What does the document say about the use of quantum computing in making cheese?", category="absent"),
    BenchmarkQuery(id="absent_02", query="Who won the Super Bowl in 2024 according to the quantum computing article?", category="absent"),
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
        resume: bool = True,
        output_file: str = "benchmark_report.json"
    ):
        self.project_id = project_id
        self.session_db = session_db
        self.user = user
        self.queries = queries or DEFAULT_BENCHMARK_QUERIES
        self.variants = variants or list(PipelineVariant)
        
        # Resumability setup
        self.partial_file = output_file.replace(".json", ".partial.json")
        self.completed_runs = set()
        self.summaries = []
        
        if resume and os.path.exists(self.partial_file):
            logger.info(f"Resuming benchmark from partial state: {self.partial_file}")
            try:
                with open(self.partial_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Reconstruct summaries
                for s_dict in data:
                    variant_val = s_dict["variant"]
                    summary = VariantSummary(
                        variant=variant_val,
                        label=s_dict["label"],
                        description=s_dict["description"]
                    )
                    
                    for qr_dict in s_dict.get("query_results", []):
                        # Construct QueryResult from dictionary
                        qr = QueryResult(**qr_dict)
                        summary.query_results.append(qr)
                        self.completed_runs.add((variant_val, qr.query_id))
                        
                    self.summaries.append(summary)
            except Exception as e:
                logger.error(f"Failed to load partial state: {e}. Starting fresh.")
                self.completed_runs = set()
                self.summaries = []
        elif not resume and os.path.exists(self.partial_file):
            logger.info("Starting fresh benchmark. Removing existing partial state.")
            os.remove(self.partial_file)
            
    def _save_partial_state(self, current_variant_summary: VariantSummary):
        """Serialize current in-progress state to disk."""
        # Find or append current summary
        found = False
        for i, s in enumerate(self.summaries):
            if s.variant == current_variant_summary.variant:
                self.summaries[i] = current_variant_summary
                found = True
                break
        if not found:
            self.summaries.append(current_variant_summary)
            
        with open(self.partial_file, "w", encoding="utf-8") as f:
            f.write(self.to_json(self.summaries))

    async def _run_single_query(
        self,
        query: BenchmarkQuery,
        variant: PipelineVariant,
        variant_config: Dict[str, Any],
        cache=None,
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
                rag_config = RAGConfig(project_id=self.project_id, is_active=True)
                self.session_db.add(rag_config)
                self.session_db.commit()
                self.session_db.refresh(rag_config)

            # Override toggles
            rag_config.use_hybrid_search = variant_config.get("use_hybrid_search", True)
            rag_config.enable_self_correction = variant_config.get("use_self_correction", True)
            rag_config.enable_adaptive_router = variant_config.get("enable_adaptive_router", False)

            if variant_config.get("enable_runtime_planner", False):
                from app.runtime.planner import RuntimePlanner
                planner = RuntimePlanner()
                st, plan = planner.plan(query.query, project_id=self.project_id)
                rag_config.top_k = plan.retrieval_top_k
                if plan.retrieval_strategy == "bm25":
                    rag_config.use_hybrid_search = False
            elif rag_config.enable_adaptive_router:
                from app.services.adaptive_retrieval_router import AdaptiveRetrievalRouter
                adaptive_router = AdaptiveRetrievalRouter()
                decision = adaptive_router.classify(query.query)
                if decision.chosen_backend == "bm25":
                    rag_config.use_hybrid_search = False

            # Build LLM client
            import os as _os
            primary_model_cfg = getattr(rag_config, "primary_models", [{"provider": "groq", "name": "openai/gpt-oss-20b"}])[0]
            provider = primary_model_cfg.get("provider", "groq")
            model_name = primary_model_cfg.get("name", "openai/gpt-oss-20b")
            
            if provider == "groq":
                from langchain_groq import ChatGroq
                llm_client = ChatGroq(
                    model_name=model_name,
                    api_key=_os.getenv("GROQ_API_KEY"),
                    temperature=0.0, # Forced for determinism in benchmarks
                )
            elif provider == "openrouter":
                from langchain_openai import ChatOpenAI
                llm_client = ChatOpenAI(
                    model_name=model_name,
                    openai_api_key=_os.getenv("OPENROUTER_API_KEY"),
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.0,
                )
            else:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm_client = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=_os.getenv("GEMINI_API_KEY"),
                    temperature=0.0, # Forced for determinism in benchmarks
                )

            try:
                search_results = engine.search(
                    query=query.query,
                    project_id=self.project_id,
                    k=rag_config.top_k,
                    score_threshold=rag_config.similarity_threshold,
                    llm_client=llm_client if variant_config.get("use_query_rewriting") else None,
                )
            except Exception as e:
                raise RuntimeError(f"Retrieval failed: {e}")

            ctx_chunks = [d.page_content for d, _ in search_results]
            result.chunks_retrieved = len(ctx_chunks)
            # Assuming Document metadata contains the chunk 'id'
            result.retrieved_chunk_ids = [d.metadata.get("chunk_id", i) for i, (d, _) in enumerate(search_results)]

            # Compute confidence from gate result
            gate_res = getattr(search_results, "confidence_gate_result", None)
            if gate_res:
                result.confidence_score = gate_res.max_confidence

            # --- Generate answer ---
            if ctx_chunks:
                context_str = "\n\n".join(ctx_chunks[:12])
                gen_prompt = (
                    "You are a precise and thorough assistant. Answer the question using ONLY information from the context below. "
                    "Be detailed and comprehensive in your response, citing specific details from the context. "
                    "Use the same terminology and phrasing found in the context whenever possible. "
                    "If the answer is not in the context, say 'I cannot find that information in the provided documents.'\n\n"
                    f"Context:\n{context_str}\n\nQuestion: {query.query}\n\nDetailed Answer:"
                )
                try:
                    if cache is not None:
                        # kb_version=1 for benchmark
                        cached = cache.get(query.query, self.project_id, 1)
                        if cached:
                            result.answer = cached
                            result.cache_hit = True
                    
                    if not result.cache_hit:
                        gen_res = await llm_client.ainvoke(gen_prompt)
                        result.answer = gen_res.content if hasattr(gen_res, "content") else str(gen_res)
                        
                        if cache is not None:
                            cache.set(query.query, result.answer, self.project_id, 1)
                except Exception as e:
                    raise RuntimeError(f"Generation failed: {e}")
            else:
                result.answer = "No relevant information found in the knowledge base."

            # --- Evaluate with RAGAS ---
            from app.services.ragas_metrics import RAGASEvaluator
            from app.services.rag_evaluator import evaluate_rag_response

            evaluator = RAGASEvaluator()
            sync_scores = evaluate_rag_response(query.query, result.answer, ctx_chunks)
            retrieved_chunks_dicts = [{"content": t, "source": ""} for t in ctx_chunks]

            # Don't pass sync_scores to RAGAS evaluator — let it use its
            # own improved metric calculations instead of the old flat ones.
            ragas = evaluator.evaluate(
                query=query.query,
                retrieved_chunks=retrieved_chunks_dicts,
                generated_answer=result.answer,
                existing_eval_scores=None,
            )

            # Debug output for Part 1a
            if query.id == "factoid_01" and variant.value in ("naive", "hybrid"):
                print(f"--- DEBUG {variant.value.upper()} ---")
                print(f"Chunks retrieved: {len(ctx_chunks)}")
                for i, c in enumerate(ctx_chunks):
                    print(f"Chunk {i}: {c[:100]}...")
                print(f"Score: {ragas.context_relevance}")

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
                model_name,
                provider,
                i_tokens,
                o_tokens,
            )

            # Cache hit detection is already done for RAGOps above
            if not result.cache_hit:
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
        # Check if we already have this summary from resume
        for s in self.summaries:
            if s.variant == variant.value:
                summary = s
                break
        else:
            summary = VariantSummary(
                variant=variant.value,
                label=config["label"],
                description=config["description"],
            )

        # Setup versioned semantic cache if the variant uses advanced planning
        from app.services.cost_control import SemanticCache
        from app.services.versioned_cache import VersionedSemanticCache
        base_cache = SemanticCache(threshold=0.85, ttl_seconds=3600)
        cache = VersionedSemanticCache(base_cache)
        enable_cache = config.get("use_semantic_cache", False)

        for query in self.queries:
            if (variant.value, query.id) in self.completed_runs:
                logger.info(f"Skipping already completed query {query.id} for variant {variant.value}")
                continue
                
            logger.info(f"Running query {query.id} for variant {variant.value}")
            result = await self._run_single_query(query, variant, config, cache if enable_cache else None)
            summary.query_results.append(result)
            self.completed_runs.add((variant.value, query.id))
            
            # Save partial state after each query
            self._save_partial_state(summary)


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
            
            # Deterministic Retrieval Metrics aggregation
            from app.benchmark.retrieval_metrics import calculate_recall_at_k, calculate_precision_at_k, calculate_mrr, calculate_ndcg
            
            gt_queries = [q for q in self.queries if getattr(q, "expected_sources", None)]
            if gt_queries:
                recalls = []
                precisions = []
                mrrs = []
                ndcgs = []
                
                for q in gt_queries:
                    # Find matching result
                    qr = next((r for r in ok_results if r.query_id == q.id), None)
                    if qr:
                        # Assuming expected_sources contains chunk IDs (ints)
                        expected = [int(s) for s in q.expected_sources]
                        retrieved = qr.retrieved_chunk_ids
                        
                        recalls.append(calculate_recall_at_k(retrieved, expected, k=5))
                        precisions.append(calculate_precision_at_k(retrieved, expected, k=5))
                        mrrs.append(calculate_mrr(retrieved, expected))
                        ndcgs.append(calculate_ndcg(retrieved, expected, k=5))
                
                if recalls:
                    summary.avg_recall_at_5 = sum(recalls) / len(recalls)
                    summary.avg_precision_at_5 = sum(precisions) / len(precisions)
                    summary.avg_mrr = sum(mrrs) / len(mrrs)
                    summary.avg_ndcg_at_5 = sum(ndcgs) / len(ndcgs)
                    
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
