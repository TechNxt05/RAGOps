from typing import List, Dict, Any, Callable
from collections import defaultdict

class MultiQueryRetriever:
    def __init__(self, llm_client, hybrid_search_fn: Callable, n_queries: int = 3):
        self.llm = llm_client
        self.hybrid_search = hybrid_search_fn
        self.n_queries = n_queries

    def generate_query_variants(self, query: str) -> List[str]:
        """
        Generate N semantically diverse paraphrases of the query.
        Uses lightweight LLM call.
        """
        prompt = f"""Generate {self.n_queries} different search queries that 
would retrieve documents relevant to answering this question. 
Make each query semantically distinct - use different words, 
perspectives, and phrasings.

Original question: {query}

Return ONLY the queries, one per line, no numbering, no explanation."""
        
        try:
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(prompt).content
            elif hasattr(self.llm, "predict"):
                response = self.llm.predict(prompt)
            elif hasattr(self.llm, "generate"):
                response = self.llm.generate(prompt)
            else:
                response = str(self.llm(prompt))
        except Exception as e:
            print(f"Error generating query variants: {e}")
            return [query]

        variants = [q.strip() for q in response.strip().split("\n") if q.strip()]
        # Filter out numbers if any (e.g. "1. query")
        cleaned_variants = []
        for v in variants:
            import re
            cleaned = re.sub(r'^\d+[\.\)\-\s]+', '', v).strip()
            if cleaned:
                cleaned_variants.append(cleaned)
                
        # Always include original query
        all_queries = [query] + cleaned_variants[:self.n_queries - 1]
        return all_queries

    def reciprocal_rank_fusion(
        self, 
        result_lists: List[List[Any]],
        k: int = 60
    ) -> List[Any]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion.
        RRF score = sum(1 / (k + rank)) across all lists.
        k=60 is the standard RRF constant.
        """
        chunk_scores = defaultdict(float)
        chunk_data = {}
        
        for result_list in result_lists:
            for rank, item in enumerate(result_list, start=1):
                # item can be a Tuple (LCDocument, float) or LCDocument
                doc = item[0] if isinstance(item, tuple) else item
                score = item[1] if isinstance(item, tuple) else 0.0
                
                meta = getattr(doc, "metadata", {}) or {}
                chunk_id = meta.get("doc_id_version") or meta.get("doc_id") or doc.page_content[:100]
                
                chunk_scores[chunk_id] += 1.0 / (k + rank)
                chunk_data[chunk_id] = (doc, score)
        
        # Sort by fused RRF score descending
        sorted_chunks = sorted(
            chunk_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        result = []
        for chunk_id, rrf_score in sorted_chunks:
            doc, orig_score = chunk_data[chunk_id]
            # We copy the document metadata to attach RRF rankings and scores
            from langchain_core.documents import Document as LCDocument
            new_meta = {**(doc.metadata or {})}
            new_meta["rrf_score"] = rrf_score
            new_meta["rrf_rank"] = len(result) + 1
            # We set the overall similarity score to the RRF score or a blend
            new_doc = LCDocument(page_content=doc.page_content, metadata=new_meta)
            result.append((new_doc, rrf_score))
        
        return result

    def retrieve(
        self, 
        query: str, 
        project_id: int,
        top_k: int = 10,
        use_multi_query: bool = True
    ) -> dict:
        """
        Main entry point. Generate query variants, retrieve for each,
        fuse with RRF, return top_k results.
        """
        if not use_multi_query:
            # Fall back to single query
            chunks = self.hybrid_search(query, project_id, top_k)
            return {
                "chunks": chunks,
                "queries_used": [query],
                "fusion_method": "single_query"
            }
        
        # Generate variants
        queries = self.generate_query_variants(query)
        
        # Retrieve for each query variant
        all_result_lists = []
        for q in queries:
            try:
                results = self.hybrid_search(q, project_id, top_k)
                all_result_lists.append(results)
            except Exception as e:
                print(f"Error in multi-query hybrid search for variant '{q}': {e}")
                continue
        
        if not all_result_lists:
            # Fallback to original query
            chunks = self.hybrid_search(query, project_id, top_k)
            return {
                "chunks": chunks,
                "queries_used": [query],
                "fusion_method": "fallback_single"
            }
        
        # Fuse results
        fused_chunks = self.reciprocal_rank_fusion(all_result_lists)
        
        return {
            "chunks": fused_chunks[:top_k],
            "queries_used": queries,
            "fusion_method": "rag_fusion_rrf",
            "query_count": len(queries),
            "total_candidates": sum(len(r) for r in all_result_lists)
        }
