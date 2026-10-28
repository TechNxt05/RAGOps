import logging
from typing import List, Tuple, Optional, Any
from sqlmodel import Session, select
from langchain_core.documents import Document as LCDocument

from app.models.rag import RAGConfig, Document
from app.services.query_rewriter import QueryRewriter
from app.services.multi_query_retriever import MultiQueryRetriever
from app.services.reranker_service import reranker_service
from app.services.confidence_gate import get_confidence_gate, ConfidenceGateResult
from app.services.ragas_metrics import RAGASEvaluator
from app.services.pipeline_tracer import PipelineStage

logger = logging.getLogger(__name__)

class SelfCorrectionPolicy:
    def __init__(self, session: Session, llm_client: Any, hybrid_search_fn: Any):
        self.session = session
        self.llm = llm_client
        self.hybrid_search = hybrid_search_fn

    async def execute_correction_loop(
        self,
        query: str,
        project_id: int,
        rag_config: RAGConfig,
        old_confidence: float,
        history_dicts: List[dict],
        tracer: Any
    ) -> Optional[Tuple[List[Tuple[LCDocument, float]], ConfidenceGateResult]]:
        """
        Executes one retry of query rewrite -> search -> rerank -> confidence gate.
        Returns (new_chunks, new_gate_result) if new confidence is better than old, else None.
        """

        if not getattr(rag_config, "enable_self_correction", True):
            logger.info("Self-correction disabled in project config.")
            return None

        logger.info(f"Self-correction loop triggered for query '{query}' with old confidence {old_confidence:.3f}")

        # Start Pipeline trace stage for SELF_CORRECTION_TRIGGERED
        tracer.start_stage(PipelineStage.SELF_CORRECTION_TRIGGERED)

        # 1. Re-write query instructing to diversify
        rewriter = QueryRewriter(self.llm)
        retry_reason = f"Low confidence retrieve relevance of {old_confidence:.3f}"
        rewrite_res = await rewriter.rewrite(query, conversation_history=history_dicts, retry_reason=retry_reason)
        retry_query = rewrite_res["rewritten_query"]
        logger.info(f"Rewritten query for retry: '{retry_query}'")

        # 2. Re-run MultiQueryRetriever + BGE Reranker
        from app.services.semantic_router import SemanticRouter
        semantic_router = SemanticRouter()
        routing_res = semantic_router.classify(retry_query, "Factoid")
        
        top_k_override = routing_res.get("top_k", rag_config.top_k or 4)
        use_mq = getattr(rag_config, "use_multi_query", True)
        rerank_top_n = routing_res.get("rerank_top_n", 3)
        candidate_k = max(top_k_override * 5, 20)

        queries_used = [retry_query]

        def _retrieve_and_score():
            from app.rag.engine import RAGEngine
            if use_mq:
                mq_retriever = MultiQueryRetriever(self.llm, self.hybrid_search, n_queries=3)
                retrieval_res = mq_retriever.retrieve(retry_query, project_id, candidate_k, use_multi_query=True)
                merged, used = retrieval_res["chunks"], retrieval_res["queries_used"]
            else:
                merged, used = self.hybrid_search(retry_query, project_id, candidate_k), [retry_query]
            reranked, scored_flag = reranker_service.rerank_scored(query, merged, rerank_top_n)
            engine = RAGEngine(self.session)
            relevance = engine.relevance_scores(query, reranked, rag_config, scored_flag)
            # Also judge each chunk against the rephrased question (it expands
            # abbreviations/jargon: "PTO" -> "paid time off") and keep the better score.
            # Only the bi-encoder cosine is used for the rephrasing: cross-encoders
            # give inflated scores to unusual LLM-written queries, while an off-topic
            # question stays off-topic under cosine however it is rephrased.
            if retry_query and retry_query.strip().lower() != query.strip().lower():
                retry_rel = engine.relevance_scores(retry_query, reranked, rag_config, False)
                relevance = [max(a, b) for a, b in zip(relevance, retry_rel)]
            scored = [(doc, rel) for (doc, _), rel in zip(reranked, relevance)]
            return scored, used, engine.evaluate_confidence(scored, rag_config)

        from starlette.concurrency import run_in_threadpool
        final_reranked, queries_used, new_gate_result = await run_in_threadpool(_retrieve_and_score)
        new_confidence = new_gate_result.max_confidence

        # Re-run RAGASEvaluator's context relevance check
        ragas_evaluator = RAGASEvaluator()
        retrieved_chunks_dicts = [{"content": doc.page_content, "source": doc.metadata.get("source", "")} for doc, _ in final_reranked]
        context_relevance_score = ragas_evaluator._compute_context_relevance(query, retrieved_chunks_dicts, None)

        logger.info(f"Self-correction loop finished. New confidence: {new_confidence:.3f}, RAGAS context relevance: {context_relevance_score:.3f}")

        # End Pipeline trace stage
        tracer.end_stage(
            PipelineStage.SELF_CORRECTION_TRIGGERED,
            status="success",
            metadata={
                "old_confidence": old_confidence,
                "new_confidence": new_confidence,
                "retry_query": retry_query,
                "ragas_context_relevance": context_relevance_score,
                "queries_used": queries_used,
                "passed": new_gate_result.passed
            }
        )

        if new_confidence > old_confidence:
            return final_reranked, new_gate_result
        else:
            return None
