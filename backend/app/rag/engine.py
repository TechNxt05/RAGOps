import os
import shutil
import logging
from typing import List, Tuple, Optional
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
        for chunk, doc in rows:
            texts.append(chunk.content)
            metadatas.append(
                {
                    "source": doc.filename,
                    "doc_id": doc.id,
                    "project_id": doc.project_id,
                }
            )

        if not texts:
            if os.path.exists(VECTOR_STORE_PATH):
                shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)
            if project_id:
                bm25_manager.delete_index(str(project_id))
            return

        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
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
            
            from app.services.adaptive_chunker import get_adaptive_chunker

            content = document.content or ""
            doc_type = document.filename.split(".")[-1] if document.filename and "." in document.filename else "unknown"
            
            chunker = get_adaptive_chunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
            texts, strategy, metrics, all_scores = chunker.select_and_chunk(
                text=content,
                document_type=doc_type
            )

            if os.path.exists(VECTOR_STORE_PATH):
                try:
                    vector_store = FAISS.load_local(
                        VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
                    )
                except Exception:
                    vector_store = None
            else:
                vector_store = None

            metadatas = [
                {"source": document.filename, "doc_id": document.id, "project_id": document.project_id}
                for _ in texts
            ]

            if not vector_store:
                vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            else:
                vector_store.add_texts(texts, metadatas=metadatas)

            vector_store.save_local(VECTOR_STORE_PATH)

            for txt in texts:
                chunk = Chunk(
                    document_id=document.id,
                    content=txt,
                    created_at=datetime.utcnow(),
                )
                self.session.add(chunk)

            document.processed = True
            document.processing_status = "complete"
            document.chunk_count = len(texts)
            document.chunk_size_used = config.chunk_size
            document.embedding_model_used = config.embedding_model or "google-embedding-001"
            
            # Save strategy metadata
            document.chunking_strategy = strategy.value
            document.chunking_metrics = {
                "composite_score": metrics.composite_score,
                "size_compliance": metrics.size_compliance,
                "intrachunk_cohesion": metrics.intrachunk_cohesion,
                "contextual_coherence": metrics.contextual_coherence,
                "block_integrity": metrics.block_integrity,
                "reference_completeness": metrics.reference_completeness,
                "all_strategy_scores": all_scores
            }
            
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

    def reindex_all(self) -> None:
        self.rebuild_full_index()

    def search(
        self, query: str, project_id: int, k: int = 4, score_threshold: float = 0.0
    ) -> List[Tuple[LCDocument, float]]:
        if not os.path.exists(VECTOR_STORE_PATH):
            return SearchResultList()
            
        from app.services.query_understanding import get_query_understanding
        from app.services.confidence_gate import get_confidence_gate
        from app.services.pipeline_tracer import PipelineTracer, PipelineStage

        tracer = PipelineTracer(query)
        
        try:
            config = self.get_active_config(project_id)
            embeddings = self._get_embeddings(config)
            
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
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
            effective_query = analysis.expanded_query
            suggested_top_k = analysis.suggested_top_k
            candidate_k = max(suggested_top_k * 5, 20)
            
            inactive = self._inactive_doc_ids(project_id)
            semantic_results: List[Tuple[LCDocument, float]] = []
            bm25_results: List[Tuple[str, float]] = []
            used_hybrid = False

            if analysis.retrieval_strategy == "multi" and len(analysis.sub_queries) > 1:
                seen_semantic = set()
                seen_bm25 = set()
                for sub_q in analysis.sub_queries:
                    # Semantic Search per sub-query
                    sub_res = vector_store.similarity_search_with_score(
                        sub_q,
                        k=candidate_k // 2,
                        filter={"project_id": project_id},
                    )
                    for doc, score in sub_res:
                        did = doc.metadata.get("doc_id")
                        if did is not None and int(did) in inactive:
                            continue
                        if doc.page_content not in seen_semantic:
                            seen_semantic.add(doc.page_content)
                            semantic_results.append((doc, score))
                    
                    # BM25 Search per sub-query
                    if config.use_hybrid_search and bm25_manager.index_exists(str(project_id)):
                        sub_bm25 = bm25_manager.search(str(project_id), sub_q, top_k=candidate_k // 2)
                        for text, score in sub_bm25:
                            if text not in seen_bm25:
                                seen_bm25.add(text)
                                bm25_results.append((text, score))
            else:
                # Single expanded query
                results_with_score = vector_store.similarity_search_with_score(
                    effective_query,
                    k=candidate_k,
                    filter={"project_id": project_id},
                )
                for doc, score in results_with_score:
                    did = doc.metadata.get("doc_id")
                    if did is not None and int(did) in inactive:
                        continue
                    semantic_results.append((doc, score))

                if config.use_hybrid_search and bm25_manager.index_exists(str(project_id)):
                    bm25_results = bm25_manager.search(str(project_id), effective_query, top_k=candidate_k)

            # Hybrid Merge (RRF)
            if config.use_hybrid_search and bm25_results:
                merged_results = hybrid_search_merge(
                    semantic_results=semantic_results,
                    bm25_results=bm25_results,
                    semantic_weight=config.semantic_weight,
                    bm25_weight=1.0 - config.semantic_weight
                )
                used_hybrid = True
            else:
                merged_results = semantic_results

            # Context Pruning
            prune_threshold = max(config.similarity_threshold or 0.0, 0.1)
            pruned_results, orig_count, pruned_count, reduction_pct = context_pruner.prune(
                query=query,
                chunks=merged_results,
                threshold=prune_threshold
            )

            # BGE Reranker
            final_reranked = reranker_service.rerank(
                query=query,
                chunks=pruned_results,
                top_k=k
            )

            tracer.end_stage(
                PipelineStage.RETRIEVAL,
                metadata={
                    "chunks_before_pruning": orig_count,
                    "chunks_after_pruning": pruned_count,
                    "pruning_reduction_pct": reduction_pct,
                    "used_hybrid_search": used_hybrid,
                    "candidate_k": candidate_k
                }
            )

            # Stage 3: Source Confidence Scoring & Gate
            tracer.start_stage(PipelineStage.CONFIDENCE_GATE)
            
            # Fetch upload date and file type for confidence gate evaluation
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
            final_results.pipeline_trace = tracer.to_dict()

            return final_results

        except Exception as e:
            logging.error(f"Error in RAGEngine.search pipeline: {e}")
            tracer.fail_pipeline(str(e))
            empty_res = SearchResultList()
            empty_res.pipeline_trace = tracer.to_dict()
            return empty_res
