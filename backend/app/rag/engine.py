import os
import shutil
import logging
from typing import List, Tuple, Optional, Any
from datetime import datetime
from sqlmodel import Session, select
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument

from app.models.rag import RAGConfig, Document, Chunk, Project
from app.services.bm25_service import bm25_manager
from app.services.rrf_service import hybrid_search_merge
from app.services.context_pruner import context_pruner
from app.services.reranker_service import reranker_service

VECTOR_STORE_PATH = "faiss_index"


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
        
        # New attributes for upgrades
        self.query_analysis = None
        self.confidence_gate_result = None
        self.pipeline_trace = None


class RAGEngine:
    def __init__(self, session: Session):
        self.session = session
        # Default fallback embeddings if no project config is found
        self._default_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

    def _get_embeddings(self, config: Optional[RAGConfig] = None):
        """Load embeddings based on config. Defaults to Google embedding-001."""
        if not config or not config.embedding_model:
            return self._default_embeddings

        if "huggingface" in config.embedding_model.lower() or "minilm" in config.embedding_model.lower():
            # Use local HuggingFace embeddings (no API cost)
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Default to Google
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

    def get_active_config(self, project_id: int) -> RAGConfig:
        config = self.session.exec(
            select(RAGConfig)
            .where(RAGConfig.project_id == project_id)
            .where(RAGConfig.is_active == True)
        ).first()
        if not config:
            raise ValueError(f"No active RAG configuration found for Project ID {project_id}")
        return config

    def _inactive_doc_ids(self, project_id: int) -> set[int]:
        rows = self.session.exec(
            select(Document.id).where(Document.project_id == project_id).where(Document.is_active == False)
        ).all()
        return {int(r) for r in rows if r is not None}

    def _rebuild_bm25_for_project(self, project_id: int) -> None:
        """Fetches all active chunks for a project and builds the BM25 index."""
        try:
            rows = self.session.exec(
                select(Chunk)
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.project_id == project_id)
                .where(Document.is_active == True)
                .where(Document.processed == True)
            ).all()
            chunks = [chunk.content for chunk in rows]
            if chunks:
                bm25_manager.build_index(str(project_id), chunks)
            else:
                bm25_manager.delete_index(str(project_id))
        except Exception as e:
            logging.error(f"Error rebuilding BM25 index for project {project_id}: {e}")

    def rebuild_full_index(self, project_id: Optional[int] = None) -> None:
        """Rebuild FAISS from all chunks belonging to active, processed documents."""
        config = None
        if project_id:
            try:
                config = self.get_active_config(project_id)
            except Exception:
                pass
        
        embeddings = self._get_embeddings(config)

        rows_query = (
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.is_active == True)
            .where(Document.processed == True)
        )
        
        if project_id:
            rows_query = rows_query.where(Document.project_id == project_id)
            
        rows = self.session.exec(rows_query).all()

        texts: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []
        
        # Keep track of chunk counts per doc to backfill chunk_index
        doc_counters = {}
        
        for chunk, doc in rows:
            texts.append(chunk.content)
            
            # Backfill Phase 2 fields if missing
            if not chunk.content_hash:
                import hashlib
                chunk.content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            if chunk.chunk_index is None:
                curr = doc_counters.get(doc.id, 0)
                chunk.chunk_index = curr
                doc_counters[doc.id] = curr + 1
            if not chunk.doc_id_version:
                chunk.doc_id_version = f"{doc.id}:{chunk.chunk_index}:{chunk.content_hash[:8]}"
            self.session.add(chunk)
            
            metadatas.append(
                {
                    "source": doc.filename,
                    "doc_id": doc.id,
                    "project_id": doc.project_id,
                    "content_hash": chunk.content_hash,
                    "doc_id_version": chunk.doc_id_version,
                }
            )
            ids.append(chunk.doc_id_version)

        self.session.commit()

        if not texts:
            if os.path.exists(VECTOR_STORE_PATH):
                shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)
            if project_id:
                bm25_manager.delete_index(str(project_id))
            return

        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas, ids=ids)
        vector_store.save_local(VECTOR_STORE_PATH)

        # Build BM25 index for projects
        if project_id:
            self._rebuild_bm25_for_project(project_id)
        else:
            projects = self.session.exec(select(Project)).all()
            for proj in projects:
                if proj.id:
                    self._rebuild_bm25_for_project(proj.id)

    def process_document(self, document: Document) -> None:
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

            embeddings = self._get_embeddings(config)
            
            # Check if this document was parsed using Docling (which pre-chunks tables and text)
            if document.parsing_method == "docling" and document.parsed_chunks_json:
                chunks_list = document.parsed_chunks_json.get("chunks", [])
                new_chunks = [
                    {
                        "index": idx,
                        "text": chunk["content"],
                        "metadata": chunk.get("metadata", {})
                    }
                    for idx, chunk in enumerate(chunks_list)
                ]
                strategy_val = "docling"
                strategy_metrics = {
                    "composite_score": 1.0,
                    "size_compliance": 1.0,
                    "intrachunk_cohesion": 1.0,
                    "contextual_coherence": 1.0,
                    "block_integrity": 1.0,
                    "reference_completeness": 1.0
                }
            else:
                # Fallback / Default Text Chunking (Adaptive)
                from app.services.adaptive_chunker import get_adaptive_chunker

                content = document.content or ""
                doc_type = document.filename.split(".")[-1] if document.filename and "." in document.filename else "unknown"
                
                chunker = get_adaptive_chunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
                texts, strategy, metrics, all_scores = chunker.select_and_chunk(
                    text=content,
                    document_type=doc_type
                )
                
                new_chunks = [
                    {
                        "index": idx,
                        "text": txt,
                        "metadata": {}
                    }
                    for idx, txt in enumerate(texts)
                ]
                strategy_val = strategy.value
                strategy_metrics = {
                    "composite_score": metrics.composite_score,
                    "size_compliance": metrics.size_compliance,
                    "intrachunk_cohesion": metrics.intrachunk_cohesion,
                    "contextual_coherence": metrics.contextual_coherence,
                    "block_integrity": metrics.block_integrity,
                    "reference_completeness": metrics.reference_completeness,
                    "all_strategy_scores": all_scores
                }

            # Initialize vector store if not exists
            if os.path.exists(VECTOR_STORE_PATH):
                try:
                    vector_store = FAISS.load_local(
                        VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
                    )
                except Exception:
                    vector_store = None
            else:
                vector_store = None

            # Execute Delta Indexing
            from app.services.delta_indexer import DeltaIndexer
            delta_indexer = DeltaIndexer(self.session, vector_store, embeddings)
            delta_stats = delta_indexer.delta_index(document.id, new_chunks)

            document.processed = True
            document.processing_status = "complete"
            document.chunk_count = len(new_chunks)
            document.chunk_size_used = config.chunk_size
            document.embedding_model_used = config.embedding_model or "google-embedding-001"
            
            # Save strategy metadata
            document.chunking_strategy = strategy_val
            document.chunking_metrics = strategy_metrics
            
            self.session.add(document)
            self.session.commit()

            # Build BM25 index for the project after processing
            self._rebuild_bm25_for_project(document.project_id)
        except Exception as exc:
            document.processed = False
            document.processing_status = "failed"
            document.processing_error = str(exc)
            self.session.add(document)
            self.session.commit()
            raise

    def _single_hybrid_search(
        self,
        query: str,
        project_id: int,
        k: int = 10,
        filter_document_id: Optional[int] = None,
        filter_sources: Optional[set[str]] = None
    ) -> List[Tuple[LCDocument, float]]:
        """
        Executes a single hybrid retrieval search (semantic + BM25 if configured),
        filtering by inactive documents, and optionally by document_id or sources.
        """
        if not os.path.exists(VECTOR_STORE_PATH):
            return []
            
        try:
            config = self.get_active_config(project_id)
        except ValueError:
            config = RAGConfig(project_id=project_id)
            
        embeddings = self._get_embeddings(config)
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        
        candidate_k = max(k * 5, 20)
        inactive = self._inactive_doc_ids(project_id)
        
        # 1. Semantic Search
        results_with_score = vector_store.similarity_search_with_score(
            query,
            k=candidate_k * 2,  # Fetch more to allow for filtering
            filter={"project_id": project_id},
        )
        
        semantic_results = []
        for doc, score in results_with_score:
            did = doc.metadata.get("doc_id")
            if did is not None and int(did) in inactive:
                continue
            if filter_document_id is not None and did != filter_document_id:
                continue
            if filter_sources is not None and doc.metadata.get("source") not in filter_sources:
                continue
            semantic_results.append((doc, score))
            
        # 2. BM25 Search
        bm25_results = []
        if config.use_hybrid_search and bm25_manager.index_exists(str(project_id)):
            raw_bm25 = bm25_manager.search(str(project_id), query, top_k=candidate_k * 2)
            # Filter BM25 results by active/inactive and Python filters
            if filter_document_id is not None or filter_sources is not None or inactive:
                # Query Chunk/Document tables to verify filter matches
                texts = [r[0] for r in raw_bm25]
                if texts:
                    db_chunks = self.session.exec(
                        select(Chunk, Document)
                        .join(Document, Chunk.document_id == Document.id)
                        .where(Document.project_id == project_id)
                        .where(Document.is_active == True)
                        .where(Chunk.content.in_(texts))
                    ).all()
                    
                    valid_texts = set()
                    for chunk, doc in db_chunks:
                        if doc.id in inactive:
                            continue
                        if filter_document_id is not None and doc.id != filter_document_id:
                            continue
                        if filter_sources is not None and doc.filename not in filter_sources:
                            continue
                        valid_texts.add(chunk.content)
                        
                    for text, score in raw_bm25:
                        if text in valid_texts:
                            bm25_results.append((text, score))
            else:
                bm25_results = raw_bm25
                
        # 3. Hybrid Merge (RRF)
        if config.use_hybrid_search and bm25_results:
            merged_results = hybrid_search_merge(
                semantic_results=semantic_results,
                bm25_results=bm25_results,
                semantic_weight=config.semantic_weight,
                bm25_weight=1.0 - config.semantic_weight
            )
        else:
            merged_results = semantic_results
            
        return merged_results[:k]

    def reindex_all(self) -> None:
        self.rebuild_full_index()

    def search(
        self, query: str, project_id: int, k: int = 4, score_threshold: float = 0.0,
        constraints=None, rewritten_query: Optional[str] = None, llm_client: Optional[Any] = None
    ) -> List[Tuple[LCDocument, float]]:
        if not os.path.exists(VECTOR_STORE_PATH):
            return SearchResultList()
            
        from app.services.query_understanding import get_query_understanding
        from app.services.confidence_gate import get_confidence_gate
        from app.services.pipeline_tracer import PipelineTracer, PipelineStage

        tracer = PipelineTracer(query)
        
        try:
            config = self.get_active_config(project_id)
            
            # Setup default LLM client if not provided
            if llm_client is None:
                if config.primary_llm_provider == "groq":
                    from langchain_groq import ChatGroq
                    llm_client = ChatGroq(
                        model_name=config.primary_llm_name,
                        api_key=os.getenv("GROQ_API_KEY"),
                        temperature=config.temperature,
                    )
                else:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm_client = ChatGoogleGenerativeAI(
                        model=config.primary_llm_name,
                        google_api_key=os.getenv("GEMINI_API_KEY"),
                        temperature=config.temperature,
                    )

            # Stage 1: Query Understanding
            tracer.start_stage(PipelineStage.QUERY_UNDERSTANDING)
            query_understander = get_query_understanding()
            analysis = query_understander.analyze(query)
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
            
            # Phase 3: Semantic Router query classification
            from app.services.semantic_router import SemanticRouter
            semantic_router = SemanticRouter()
            routing_res = semantic_router.classify(query, analysis.complexity.value)
            
            top_k_override = routing_res["top_k"]
            use_mq = routing_res["use_multi_query"]
            rerank_top_n = routing_res["rerank_top_n"]
            candidate_k = max(top_k_override * 5, 20)
            
            queries_used = [effective_query]
            multi_query_info = None

            # Execute multi-query if enabled by SemanticRouter
            if use_mq:
                from app.services.multi_query_retriever import MultiQueryRetriever
                mq_retriever = MultiQueryRetriever(llm_client, self._single_hybrid_search, n_queries=3)
                retrieval_res = mq_retriever.retrieve(effective_query, project_id, candidate_k, use_multi_query=True)
                merged_results = retrieval_res["chunks"]
                queries_used = retrieval_res["queries_used"]
                multi_query_info = {
                    "queries_used": queries_used,
                    "fusion_method": retrieval_res["fusion_method"],
                    "query_count": len(queries_used),
                    "total_candidates": retrieval_res["total_candidates"]
                }
            elif analysis.retrieval_strategy == "multi" and len(analysis.sub_queries) > 1:
                seen_contents = set()
                merged_results = []
                for sub_q in analysis.sub_queries:
                    sub_res = self._single_hybrid_search(sub_q, project_id, candidate_k // 2)
                    for doc, score in sub_res:
                        if doc.page_content not in seen_contents:
                            seen_contents.add(doc.page_content)
                            merged_results.append((doc, score))
            else:
                merged_results = self._single_hybrid_search(effective_query, project_id, candidate_k)

            used_hybrid = getattr(config, "use_hybrid_search", True)

            # Apply Hard Constraints
            if constraints and constraints.has_constraints:
                from app.services.constraint_extractor import ConstraintExtractor
                extractor = ConstraintExtractor()
                dict_chunks = [{"id": "", "text": doc.page_content, "source": doc.metadata.get("source", ""), "_orig_doc": doc, "_orig_score": score} for doc, score in merged_results]
                filtered_dicts = extractor.apply_to_chunks(dict_chunks, constraints)
                if filtered_dicts:
                    merged_results = [(d["_orig_doc"], d["_orig_score"]) for d in filtered_dicts]

            # Context Pruning (Old TF-IDF pruner bypassed/routed around)
            pruned_results = merged_results
            orig_count = len(merged_results)
            pruned_count = len(merged_results)
            reduction_pct = 0.0

            # Cross-Reference Resolution
            from app.services.cross_reference_resolver import CrossReferenceResolver
            xref_resolver = CrossReferenceResolver()
            resolver_res = xref_resolver.resolve_all(
                chunks=pruned_results,
                hybrid_search_fn=self._single_hybrid_search,
                project_id=project_id,
                max_resolutions=3
            )
            additional_chunks = resolver_res["additional_chunks"]
            if additional_chunks:
                pruned_results = pruned_results + additional_chunks

            # BGE Reranker using rerank_top_n from routing
            final_reranked = reranker_service.rerank(
                query=query,
                chunks=pruned_results,
                top_k=rerank_top_n
            )

            # Conflict Detection
            from app.services.conflict_detector import ConflictDetector
            conflict_detector = ConflictDetector()
            conflict_res = conflict_detector.detect_conflicts(final_reranked, query)

            # Phase 3: Contextual Compressor sentence-level compression
            from app.services.contextual_compressor import ContextualCompressor
            compressor = ContextualCompressor()
            chunks_dicts = []
            for doc, score in final_reranked:
                chunks_dicts.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "doc_id": doc.metadata.get("doc_id"),
                    "score": score,
                    "metadata": doc.metadata,
                    "_orig_doc": doc
                })
            compression_res = compressor.compress_chunks(
                query=effective_query,
                chunks=chunks_dicts,
                max_total_tokens=2000,
                min_sentences_per_chunk=1
            )
            compressed_reranked = []
            for c in compression_res["compressed_chunks"]:
                orig_doc = c["_orig_doc"]
                new_doc = LCDocument(
                    page_content=c["content"],
                    metadata={
                        **orig_doc.metadata,
                        "compression_applied": True,
                        "sentences_kept": c["sentences_kept"],
                        "sentences_dropped": c["sentences_dropped"],
                        "original_content": c["original_content"]
                    }
                )
                compressed_reranked.append((new_doc, c["score"]))
            final_reranked = compressed_reranked

            retrieval_metadata = {
                "chunks_before_pruning": orig_count,
                "chunks_after_pruning": pruned_count,
                "pruning_reduction_pct": reduction_pct,
                "used_hybrid_search": used_hybrid,
                "candidate_k": candidate_k
            }
            if multi_query_info:
                retrieval_metadata["multi_query_info"] = multi_query_info
            
            retrieval_metadata["cross_reference_resolution"] = {
                "references_found": resolver_res["references_found"],
                "references_resolved": resolver_res["references_resolved"],
                "details": resolver_res["reference_details"]
            }
            retrieval_metadata["conflict_detection"] = conflict_res

            tracer.end_stage(
                PipelineStage.RETRIEVAL,
                metadata=retrieval_metadata
            )

            # Stage 3: Source Confidence Scoring & Gate
            tracer.start_stage(PipelineStage.CONFIDENCE_GATE)
            
            doc_ids = list(set(int(doc.metadata.get("doc_id")) for doc, _ in final_reranked if doc.metadata.get("doc_id") is not None))
            doc_meta_map = {}
            if doc_ids:
                docs = self.session.exec(select(Document).where(Document.id.in_(doc_ids))).all()
                for d in docs:
                    doc_meta_map[d.id] = {
                        "file_type": d.filename.split(".")[-1] if "." in d.filename else "unknown",
                        "upload_date": d.uploaded_at
                    }

            chunk_dicts = []
            reranker_scores = []
            document_metadata = []
            for doc, score in final_reranked:
                did = doc.metadata.get("doc_id")
                meta = doc_meta_map.get(did, {}) if did else {}
                chunk_dicts.append({
                    "id": str(did or ""),
                    "text": doc.page_content
                })
                reranker_scores.append(score)
                document_metadata.append(meta)

            confidence_gate = get_confidence_gate(threshold=0.65)
            gate_result = confidence_gate.evaluate(chunk_dicts, reranker_scores, document_metadata)
            
            tracer.end_stage(
                PipelineStage.CONFIDENCE_GATE,
                status="success" if gate_result.passed else "failed",
                metadata={
                    "passed": gate_result.passed,
                    "max_confidence": gate_result.max_confidence,
                    "avg_confidence": gate_result.avg_confidence,
                    "low_confidence_chunks": gate_result.low_confidence_chunks,
                    "refusal_reason": gate_result.refusal_reason
                },
                error=gate_result.refusal_reason if not gate_result.passed else None
            )

            # Package output
            tracer.complete_pipeline()
            final_results = SearchResultList(final_reranked)
            final_results.chunks_before_pruning = orig_count
            final_results.chunks_after_pruning = pruned_count
            final_results.pruning_reduction_pct = reduction_pct
            final_results.used_hybrid_search = used_hybrid
            final_results.query_analysis = analysis
            final_results.confidence_gate_result = gate_result
            final_results.conflict_detection = conflict_res
            final_results.pipeline_trace = tracer.to_dict()
            final_results.pipeline_trace["semantic_routing"] = routing_res
            final_results.pipeline_trace["compression_stats"] = {
                "original_token_estimate": compression_res["original_token_estimate"],
                "compressed_token_estimate": compression_res["compressed_token_estimate"],
                "compression_ratio": compression_res["compression_ratio"],
                "sentences_kept": compression_res["sentences_kept"],
                "sentences_dropped": compression_res["sentences_dropped"]
            }

            return final_results


        except Exception as e:
            logging.error(f"Error in RAGEngine.search pipeline: {e}")
            tracer.fail_pipeline(str(e))
            empty_res = SearchResultList()
            empty_res.pipeline_trace = tracer.to_dict()
            return empty_res
