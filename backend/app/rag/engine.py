import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document as LCDocument
from sqlalchemy import update
from sqlmodel import Session, select

from app.models.rag import Chunk, Document, Project, RAGConfig
from app.rag import vector_store
from app.services import model_registry
from app.services.bm25_service import bm25_manager
from app.services.reranker_service import reranker_service
from app.services.rrf_service import hybrid_search_merge

logger = logging.getLogger(__name__)


class SearchResultList(list):
    """
    Subclass of list to hold context pruning and hybrid retrieval metadata,
    allowing existing routing layers to access these statistics without breaking
    signature compatibility.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunks_before_pruning = 0
        self.chunks_after_pruning = 0
        self.pruning_reduction_pct = 0.0
        self.used_hybrid_search = False

        self.query_analysis = None
        self.confidence_gate_result = None
        self.conflict_detection = None
        self.pipeline_trace = None


def bump_kb_version(session: Session, project_id: int) -> None:
    """Atomically advance the project's knowledge-base version. Invalidates the
    BM25 cache and the semantic answer cache for the project."""
    session.exec(
        update(Project)
        .where(Project.id == project_id)
        .values(kb_version=Project.kb_version + 1, kb_version_updated_at=datetime.utcnow())
    )
    bm25_manager.invalidate(project_id)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RAGEngine:
    def __init__(self, session: Session):
        self.session = session
        self._configs: Dict[int, RAGConfig] = {}

    # ------------------------------------------------------------------ config
    def _get_embeddings(self, config: Optional[RAGConfig] = None):
        return model_registry.get_embedder(config.embedding_model if config else None)

    def _embedding_model(self, config: Optional[RAGConfig]) -> str:
        return model_registry.normalize_model_name(config.embedding_model if config else None)

    def get_active_config(self, project_id: int) -> RAGConfig:
        cached = self._configs.get(project_id)
        if cached is not None:
            return cached
        config = self.session.exec(
            select(RAGConfig)
            .where(RAGConfig.project_id == project_id)
            .where(RAGConfig.is_active == True)  # noqa: E712
            .order_by(RAGConfig.created_at.desc())
        ).first()
        if not config:
            raise ValueError(f"No active RAG configuration found for Project ID {project_id}")
        self._configs[project_id] = config
        return config

    def _config_or_default(self, project_id: int) -> RAGConfig:
        try:
            return self.get_active_config(project_id)
        except ValueError:
            return RAGConfig(project_id=project_id)

    def _kb_version(self, project_id: int) -> int:
        project = self.session.get(Project, project_id)
        return (project.kb_version or 1) if project else 1

    def _inactive_doc_ids(self, project_id: int) -> set[int]:
        rows = self.session.exec(
            select(Document.id).where(Document.project_id == project_id).where(Document.is_active == False)  # noqa: E712
        ).all()
        return {int(r) for r in rows if r is not None}

    # --------------------------------------------------------------- indexing
    def _chunk_document(self, document: Document, config: RAGConfig) -> Tuple[List[dict], str, dict]:
        if document.parsing_method == "docling" and document.parsed_chunks_json:
            chunks_list = document.parsed_chunks_json.get("chunks", [])
            new_chunks = [
                {"text": c["content"], "metadata": c.get("metadata", {})}
                for c in chunks_list if c.get("content", "").strip()
            ]
            metrics = {
                "composite_score": 1.0, "size_compliance": 1.0, "intrachunk_cohesion": 1.0,
                "contextual_coherence": 1.0, "block_integrity": 1.0, "reference_completeness": 1.0,
            }
            return new_chunks, "docling", metrics

        from app.services.adaptive_chunker import get_adaptive_chunker
        content = document.content or ""
        doc_type = document.filename.rsplit(".", 1)[-1] if document.filename and "." in document.filename else "unknown"
        chunker = get_adaptive_chunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        texts, strategy, metrics, all_scores = chunker.select_and_chunk(text=content, document_type=doc_type)
        new_chunks = [{"text": t, "metadata": {}} for t in texts if t and t.strip()]
        strategy_metrics = {
            "composite_score": metrics.composite_score,
            "size_compliance": metrics.size_compliance,
            "intrachunk_cohesion": metrics.intrachunk_cohesion,
            "contextual_coherence": metrics.contextual_coherence,
            "block_integrity": metrics.block_integrity,
            "reference_completeness": metrics.reference_completeness,
            "all_strategy_scores": all_scores,
        }
        return new_chunks, strategy.value, strategy_metrics

    def _sync_chunks(self, document: Document, new_chunks: List[dict], model: str) -> dict:
        """Content-addressed delta indexing: chunks whose text is unchanged keep
        their row and their embedding (even if they moved position); only new or
        edited text is embedded; vanished chunks are deleted. One transaction."""
        existing = self.session.exec(select(Chunk).where(Chunk.document_id == document.id)).all()
        by_hash: Dict[str, List[Chunk]] = {}
        for ch in existing:
            ch_hash = ch.content_hash or _sha256(ch.content)
            by_hash.setdefault(ch_hash, []).append(ch)

        kept: List[Chunk] = []
        created: List[Chunk] = []
        for idx, item in enumerate(new_chunks):
            text = item["text"]
            ch_hash = _sha256(text)
            bucket = by_hash.get(ch_hash)
            if bucket:
                chunk = bucket.pop()
                kept.append(chunk)
            else:
                chunk = Chunk(document_id=document.id, content=text, content_hash=ch_hash, chunk_version=1)
                created.append(chunk)
            chunk.chunk_index = idx
            chunk.content_hash = ch_hash
            chunk.doc_id_version = f"{document.id}:{idx}:{ch_hash[:8]}"
            chunk.chunk_metadata = item.get("metadata") or None
            self.session.add(chunk)

        stale = [c for bucket in by_hash.values() for c in bucket]
        stale_ids = [c.id for c in stale if c.id is not None]
        vector_store.delete_for_chunks(self.session, stale_ids)
        for c in stale:
            self.session.delete(c)
        self.session.flush()

        all_chunks = kept + created
        have = vector_store.embedded_chunk_ids(self.session, [c.id for c in all_chunks], model)
        to_embed = [c for c in all_chunks if c.id not in have]
        if to_embed:
            vectors = model_registry.embed_documents([c.content for c in to_embed], model)
            vector_store.add_embeddings(
                self.session,
                [(c.id, document.id, document.project_id) for c in to_embed],
                vectors,
                model,
            )
            vector_store.ensure_hnsw_index(len(vectors[0]))

        return {
            "total_chunks": len(new_chunks),
            "added": len(created),
            "updated": 0,
            "deleted": len(stale),
            "unchanged": len(kept),
            "embedded": len(to_embed),
            "indexed_at": datetime.utcnow().isoformat(),
        }

    def process_document(self, document: Document) -> dict:
        if not document.project_id:
            raise ValueError("Document must have a project_id")

        document.processing_status = "processing"
        document.processing_error = None
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)

        try:
            try:
                config = self.get_active_config(document.project_id)
            except ValueError:
                config = RAGConfig(project_id=document.project_id)
                self.session.add(config)
                self.session.commit()
                self.session.refresh(config)
                self._configs[document.project_id] = config

            model = self._embedding_model(config)
            new_chunks, strategy_val, strategy_metrics = self._chunk_document(document, config)
            stats = self._sync_chunks(document, new_chunks, model)

            document.processed = True
            document.processing_status = "complete"
            document.chunk_count = len(new_chunks)
            document.chunk_size_used = config.chunk_size
            document.embedding_model_used = model
            document.chunking_strategy = strategy_val
            document.chunking_metrics = strategy_metrics
            self.session.add(document)
            bump_kb_version(self.session, document.project_id)
            self.session.commit()
            return stats
        except Exception as exc:
            self.session.rollback()
            document.processed = False
            document.processing_status = "failed"
            document.processing_error = str(exc)[:2000]
            self.session.add(document)
            self.session.commit()
            raise

    def rebuild_full_index(self, project_id: Optional[int] = None) -> None:
        """(Re-)embed every chunk of a project with its configured embedding model.
        Only the given project is touched; with no project_id, every project is
        rebuilt one at a time."""
        if project_id is None:
            for pid in self.session.exec(select(Project.id)).all():
                self.rebuild_full_index(pid)
            return

        config = self._config_or_default(project_id)
        model = self._embedding_model(config)
        rows = self.session.exec(
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.project_id == project_id)
            .where(Document.is_active == True)  # noqa: E712
        ).all()

        for chunk, doc in rows:
            if not chunk.content_hash:
                chunk.content_hash = _sha256(chunk.content)
                self.session.add(chunk)

        vector_store.delete_other_models(self.session, project_id, model)
        chunk_ids = [c.id for c, _ in rows]
        have = vector_store.embedded_chunk_ids(self.session, chunk_ids, model)
        missing = [(c, d) for c, d in rows if c.id not in have]
        for start in range(0, len(missing), 256):
            batch = missing[start:start + 256]
            vectors = model_registry.embed_documents([c.content for c, _ in batch], model)
            vector_store.add_embeddings(self.session, [(c.id, d.id, project_id) for c, d in batch], vectors, model)
            vector_store.ensure_hnsw_index(len(vectors[0]))
            self.session.flush()
        bump_kb_version(self.session, project_id)
        self.session.commit()

    def reindex_all(self) -> None:
        self.rebuild_full_index()

    # -------------------------------------------------------------- retrieval
    def _load_docs(self, scored: List[Tuple[int, float]], score_key: str) -> List[Tuple[LCDocument, float]]:
        if not scored:
            return []
        ids = [cid for cid, _ in scored]
        rows = self.session.exec(
            select(Chunk, Document).join(Document, Chunk.document_id == Document.id).where(Chunk.id.in_(ids))
        ).all()
        by_id = {c.id: (c, d) for c, d in rows}
        out = []
        for cid, score in scored:
            if cid not in by_id:
                continue
            chunk, doc = by_id[cid]
            meta = dict(chunk.chunk_metadata or {})
            meta.update({
                "source": doc.filename,
                "doc_id": doc.id,
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "project_id": doc.project_id,
                "content_hash": chunk.content_hash,
                "doc_id_version": chunk.doc_id_version,
                score_key: round(float(score), 6),
            })
            out.append((LCDocument(page_content=chunk.content, metadata=meta), float(score)))
        return out

    def semantic_search(
        self, query: str, project_id: int, k: int,
        filter_document_id: Optional[int] = None, filter_sources: Optional[Set[str]] = None,
    ) -> List[Tuple[LCDocument, float]]:
        config = self._config_or_default(project_id)
        model = self._embedding_model(config)
        query_vec = model_registry.embed_query(query, model)
        hits = vector_store.search(
            project_id, query_vec, model, k,
            document_id=filter_document_id, sources=filter_sources,
            exact=self._use_exact_search(project_id, model),
        )
        return self._load_docs(hits, "vector_score")

    def keyword_search(
        self, query: str, project_id: int, k: int,
        filter_document_id: Optional[int] = None, filter_sources: Optional[Set[str]] = None,
    ) -> List[Tuple[LCDocument, float]]:
        hits = bm25_manager.search(
            self.session, project_id, self._kb_version(project_id), query, top_k=k,
            document_id=filter_document_id, sources=filter_sources,
        )
        return self._load_docs(hits, "bm25_score")

    _exact_cache: Dict[Tuple[int, str, int], bool] = {}

    def _use_exact_search(self, project_id: int, model: str) -> bool:
        key = (project_id, model, self._kb_version(project_id))
        if key not in self._exact_cache:
            self._exact_cache[key] = vector_store.count_vectors(project_id, model) <= vector_store.EXACT_SEARCH_LIMIT
        return self._exact_cache[key]

    def _single_hybrid_search(
        self,
        query: str,
        project_id: int,
        k: int = 10,
        filter_document_id: Optional[int] = None,
        filter_sources: Optional[set[str]] = None
    ) -> List[Tuple[LCDocument, float]]:
        """One retrieval pass: vector search, plus BM25 fused with weighted RRF
        when hybrid search is enabled for the project."""
        config = self._config_or_default(project_id)
        candidate_k = max(k * 5, 20)

        semantic_results = self.semantic_search(query, project_id, candidate_k * 2, filter_document_id, filter_sources)
        if not config.use_hybrid_search:
            return semantic_results[:k]

        bm25_results = self.keyword_search(query, project_id, candidate_k * 2, filter_document_id, filter_sources)
        if not bm25_results:
            return semantic_results[:k]
        merged = hybrid_search_merge(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            semantic_weight=config.semantic_weight,
            bm25_weight=1.0 - config.semantic_weight,
        )
        return merged[:k]

    def relevance_scores(self, query: str, docs: List[Tuple[LCDocument, float]], config: RAGConfig, reranked: bool) -> List[float]:
        """Calibrated relevance in [0, 1] for the confidence gate.

        Two independent estimators are combined with max(): the cross-encoder
        (precise, but can miss heavy paraphrase - "signs off on overseas trips"
        vs "international travel must be approved") and bi-encoder cosine (robust
        to paraphrase). On labelled pairs neither produced a false positive
        (irrelevant <= 0.013 on both), so max() rescues paraphrases without
        letting unrelated chunks through. Ranking order is unaffected."""
        if not docs:
            return []
        model = self._embedding_model(config)
        q = model_registry.embed_query(query, model)
        # Reuse the chunk vectors stored at indexing time; embed only what's missing
        # (e.g. text altered by compression or synthetic computation results).
        stored = vector_store.stored_vectors(
            [int(d.metadata["chunk_id"]) for d, _ in docs
             if d.metadata.get("chunk_id") is not None and not d.metadata.get("compression_applied")],
            model,
        )
        vecs = [stored.get(d.metadata.get("chunk_id")) if not d.metadata.get("compression_applied") else None for d, _ in docs]
        missing = [i for i, v in enumerate(vecs) if v is None]
        if missing:
            fresh = model_registry.embed_documents([docs[i][0].page_content for i in missing], model)
            for i, v in zip(missing, fresh):
                vecs[i] = v
        cos_rel = [model_registry.cosine_to_relevance(model_registry.cosine(q, v)) for v in vecs]
        if not reranked:
            return cos_rel
        return [max(float(s), c) for (_, s), c in zip(docs, cos_rel)]

    def search(
        self, query: str, project_id: int, k: int = 4, score_threshold: float = 0.0,
        constraints=None, rewritten_query: Optional[str] = None, llm_client: Optional[Any] = None
    ) -> List[Tuple[LCDocument, float]]:
        from app.services.query_understanding import get_query_understanding
        from app.services.confidence_gate import get_confidence_gate
        from app.services.pipeline_tracer import PipelineTracer, PipelineStage

        tracer = PipelineTracer(query)

        try:
            config = self._config_or_default(project_id)
            if llm_client is None:
                from app.services.llm_factory import build_chat_model, model_chain
                primary = model_chain(config)[0]
                llm_client = build_chat_model(primary["provider"], primary["name"], temperature=config.temperature)

            # Stage 1: Query Understanding
            tracer.start_stage(PipelineStage.QUERY_UNDERSTANDING)
            analysis = get_query_understanding().analyze(query)
            tracer.end_stage(
                PipelineStage.QUERY_UNDERSTANDING,
                metadata={
                    "complexity": analysis.complexity.value,
                    "sub_queries": analysis.sub_queries,
                    "expanded_query": analysis.expanded_query,
                    "suggested_top_k": analysis.suggested_top_k,
                    "confidence": analysis.confidence
                }
            )

            # Stage 2: Retrieval
            tracer.start_stage(PipelineStage.RETRIEVAL)
            effective_query = rewritten_query if rewritten_query else analysis.expanded_query

            enable_hyde = bool(getattr(config, "enable_hyde", False)) or config.pipeline_variant == "hyde"
            hyde_retriever = None
            if enable_hyde:
                from app.services.hyde_retriever import HyDERetriever
                hyde_retriever = HyDERetriever(llm_client)
                search_query = hyde_retriever.generate_hypothetical_document(effective_query)
            else:
                search_query = effective_query

            from app.services.semantic_router import SemanticRouter
            routing_res = SemanticRouter().classify(query, analysis.complexity.value)

            # With the adaptive router on, the router sizes retrieval per query;
            # otherwise the project's configured top_k is authoritative.
            if getattr(config, "enable_adaptive_router", True):
                final_k = max(int(routing_res["rerank_top_n"]), 1)
                use_mq = bool(routing_res["use_multi_query"]) and bool(getattr(config, "use_multi_query", True))
            else:
                final_k = max(int(k or config.top_k or 4), 1)
                use_mq = bool(getattr(config, "use_multi_query", True))
            candidate_k = max(final_k * 5, 20)

            multi_query_info = None
            if use_mq:
                from app.services.multi_query_retriever import MultiQueryRetriever
                mq_retriever = MultiQueryRetriever(llm_client, self._single_hybrid_search, n_queries=3)
                retrieval_res = mq_retriever.retrieve(search_query, project_id, candidate_k, use_multi_query=True)
                merged_results = retrieval_res["chunks"]
                multi_query_info = {
                    "queries_used": retrieval_res["queries_used"],
                    "fusion_method": retrieval_res["fusion_method"],
                    "query_count": len(retrieval_res["queries_used"]),
                    "total_candidates": retrieval_res["total_candidates"]
                }
            elif analysis.retrieval_strategy == "multi" and len(analysis.sub_queries) > 1:
                seen: Set[Any] = set()
                merged_results = []
                for sub_q in analysis.sub_queries:
                    sub_search = hyde_retriever.generate_hypothetical_document(sub_q) if hyde_retriever else sub_q
                    for doc, score in self._single_hybrid_search(sub_search, project_id, max(candidate_k // 2, 10)):
                        key = doc.metadata.get("chunk_id") or doc.page_content
                        if key not in seen:
                            seen.add(key)
                            merged_results.append((doc, score))
            else:
                merged_results = self._single_hybrid_search(search_query, project_id, candidate_k)

            used_hybrid = bool(getattr(config, "use_hybrid_search", True))

            if constraints and constraints.has_constraints:
                from app.services.constraint_extractor import ConstraintExtractor
                dict_chunks = [{"id": "", "text": d.page_content, "source": d.metadata.get("source", ""), "_orig_doc": d, "_orig_score": s} for d, s in merged_results]
                filtered = ConstraintExtractor().apply_to_chunks(dict_chunks, constraints)
                if filtered:
                    merged_results = [(d["_orig_doc"], d["_orig_score"]) for d in filtered]

            orig_count = len(merged_results)
            candidates = merged_results

            resolver_res = {"references_found": 0, "references_resolved": 0, "reference_details": [], "additional_chunks": []}
            if getattr(config, "enable_cross_reference", True):
                from app.services.cross_reference_resolver import CrossReferenceResolver
                resolver_res = CrossReferenceResolver().resolve_all(
                    chunks=candidates,
                    hybrid_search_fn=self._single_hybrid_search,
                    project_id=project_id,
                    max_resolutions=3
                )
                if resolver_res["additional_chunks"]:
                    candidates = candidates + resolver_res["additional_chunks"]

            reranked = False
            if getattr(config, "enable_reranker", True):
                final_ranked, reranked = reranker_service.rerank_scored(query, candidates, final_k)
            else:
                final_ranked = candidates[:final_k]

            relevance = self.relevance_scores(query, final_ranked, config, reranked)
            final_ranked = [(d, rel) for (d, _), rel in zip(final_ranked, relevance)]
            threshold = max(float(score_threshold or 0.0), float(config.similarity_threshold or 0.0))
            if threshold > 0:
                kept = [(d, s) for d, s in final_ranked if s >= threshold]
                final_ranked = kept or final_ranked[:1]

            conflict_res = None
            if getattr(config, "enable_conflict_detection", True):
                from app.services.conflict_detector import ConflictDetector
                conflict_res = ConflictDetector().detect_conflicts(final_ranked, query)

            compression_stats = None
            if getattr(config, "enable_compression", True) and final_ranked:
                from app.services.contextual_compressor import ContextualCompressor
                chunks_dicts = [{
                    "content": d.page_content, "source": d.metadata.get("source", ""),
                    "doc_id": d.metadata.get("doc_id"), "score": s, "metadata": d.metadata, "_orig_doc": d,
                } for d, s in final_ranked]
                compression_res = ContextualCompressor().compress_chunks(
                    query=effective_query,
                    chunks=chunks_dicts,
                    max_total_tokens=max(int(config.max_context_tokens or 2048), 256),
                    min_sentences_per_chunk=1
                )
                compressed = []
                for c in compression_res["compressed_chunks"]:
                    orig = c["_orig_doc"]
                    compressed.append((LCDocument(page_content=c["content"], metadata={
                        **orig.metadata,
                        "compression_applied": c["compression_applied"],
                        "sentences_kept": c["sentences_kept"],
                        "sentences_dropped": c["sentences_dropped"],
                        "original_content": c["original_content"],
                    }), c["score"]))
                final_ranked = compressed
                compression_stats = {
                    "original_token_estimate": compression_res["original_token_estimate"],
                    "compressed_token_estimate": compression_res["compressed_token_estimate"],
                    "compression_ratio": compression_res["compression_ratio"],
                    "sentences_kept": compression_res["sentences_kept"],
                    "sentences_dropped": compression_res["sentences_dropped"],
                }

            pruned_count = len(final_ranked)
            reduction_pct = round(100.0 * (1 - pruned_count / orig_count), 2) if orig_count else 0.0
            retrieval_metadata = {
                "chunks_before_pruning": orig_count,
                "chunks_after_pruning": pruned_count,
                "pruning_reduction_pct": reduction_pct,
                "used_hybrid_search": used_hybrid,
                "candidate_k": candidate_k,
                "final_k": final_k,
                "reranked": reranked,
                "cross_reference_resolution": {
                    "references_found": resolver_res["references_found"],
                    "references_resolved": resolver_res["references_resolved"],
                    "details": resolver_res["reference_details"],
                },
                "conflict_detection": conflict_res,
            }
            if multi_query_info:
                retrieval_metadata["multi_query_info"] = multi_query_info
            tracer.end_stage(PipelineStage.RETRIEVAL, metadata=retrieval_metadata)

            # Stage 3: Source Confidence Gate
            tracer.start_stage(PipelineStage.CONFIDENCE_GATE)
            gate_result = self.evaluate_confidence(final_ranked, config)
            tracer.end_stage(
                PipelineStage.CONFIDENCE_GATE,
                status="success" if gate_result.passed else "failed",
                metadata={
                    "passed": gate_result.passed,
                    "threshold": getattr(config, "confidence_threshold", 0.4),
                    "max_confidence": gate_result.max_confidence,
                    "avg_confidence": gate_result.avg_confidence,
                    "low_confidence_chunks": gate_result.low_confidence_chunks,
                    "refusal_reason": gate_result.refusal_reason
                },
                error=gate_result.refusal_reason if not gate_result.passed else None
            )

            tracer.complete_pipeline()
            final_results = SearchResultList(final_ranked)
            final_results.chunks_before_pruning = orig_count
            final_results.chunks_after_pruning = pruned_count
            final_results.pruning_reduction_pct = reduction_pct
            final_results.used_hybrid_search = used_hybrid
            final_results.query_analysis = analysis
            final_results.confidence_gate_result = gate_result
            final_results.conflict_detection = conflict_res
            final_results.pipeline_trace = tracer.to_dict()
            final_results.pipeline_trace["semantic_routing"] = routing_res
            if compression_stats:
                final_results.pipeline_trace["compression_stats"] = compression_stats
            return final_results

        except Exception as e:
            logger.error(f"Error in RAGEngine.search pipeline: {e}", exc_info=True)
            tracer.fail_pipeline(str(e))
            empty_res = SearchResultList()
            empty_res.pipeline_trace = tracer.to_dict()
            return empty_res

    def evaluate_confidence(self, ranked: List[Tuple[LCDocument, float]], config: Optional[RAGConfig] = None):
        """Run the source-confidence gate over (doc, relevance) pairs."""
        from app.services.confidence_gate import get_confidence_gate
        doc_ids = list({int(d.metadata["doc_id"]) for d, _ in ranked if d.metadata.get("doc_id") is not None})
        doc_meta: Dict[int, dict] = {}
        if doc_ids:
            for d in self.session.exec(select(Document).where(Document.id.in_(doc_ids))).all():
                doc_meta[d.id] = {
                    "file_type": d.filename.rsplit(".", 1)[-1] if "." in d.filename else "unknown",
                    "upload_date": d.uploaded_at,
                }
        chunk_dicts, scores, metas = [], [], []
        for doc, score in ranked:
            did = doc.metadata.get("doc_id")
            chunk_dicts.append({"id": str(did or ""), "text": doc.page_content})
            scores.append(score)
            metas.append(doc_meta.get(did, {}) if did else {})
        threshold = float(getattr(config, "confidence_threshold", None) or 0.4)
        return get_confidence_gate(threshold=threshold).evaluate(chunk_dicts, scores, metas)
