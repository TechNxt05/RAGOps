import time
from typing import List, TypedDict, Optional
from sqlmodel import select

from langgraph.graph import StateGraph, END

# Import existing services & helpers
from app.services.bm25_service import bm25_manager
from app.services.rrf_service import hybrid_search_merge
from app.services.context_pruner import context_pruner
from app.services.reranker_service import reranker_service
from app.services.confidence_gate import get_confidence_gate
from app.services.query_understanding import get_query_understanding
from app.rag.engine import RAGEngine

class AgentState(TypedDict, total=False):
    query: str
    query_type: str                     # from existing Query Understanding
    project_id: str
    attempt_count: int                  # starts at 0
    strategies_tried: List[str]         # track what was tried
    current_strategy: str               # "semantic" | "hybrid" | "decomposed"
    retrieval_results: List[dict]       # raw chunks from retrieval
    confidence_score: float             # from existing confidence gate
    context: str                        # pruned context (TF-IDF)
    response: str                       # final LLM output
    agent_trace: List[dict]             # per-step trace for UI
    answered: bool
    sub_queries: List[str]
    # Temporary fields for flow communication
    _temp_latency: float
    _temp_gate_passed: bool
    _temp_refusal_reason: str


def plan_retrieval(state: AgentState) -> dict:
    """Reads query complexity type and decides initial strategy."""
    query = state["query"]
    query_understander = get_query_understanding()
    analysis = query_understander.analyze(query)
    
    complexity = analysis.complexity.value
    
    # Map complexity to initial strategy
    if complexity == "factoid":
        initial_strategy = "semantic"
    elif complexity == "analytical":
        initial_strategy = "hybrid"
    elif complexity == "multi_hop":
        initial_strategy = "decomposed"
    else:
        initial_strategy = "semantic"
        
    return {
        "query_type": complexity,
        "current_strategy": initial_strategy,
        "attempt_count": 0,
        "strategies_tried": [],
        "agent_trace": [],
        "sub_queries": getattr(analysis, "sub_queries", [query]),
        "answered": False
    }


def execute_retrieval(state: AgentState, config: dict = None) -> dict:
    """Executes the chosen retrieval strategy using existing pipeline components."""
    start_time = time.time()
    
    # Extract session from config
    configurable = (config or {}).get("configurable", {})
    session = configurable.get("session")
    if not session:
        raise ValueError("SQLModel Session is required in LangGraph configurable context.")
        
    query = state["query"]
    project_id = int(state["project_id"])
    strategy = state["current_strategy"]
    sub_queries = state.get("sub_queries", [query])
    attempt = state["attempt_count"] + 1
    
    # Initialize RAG Engine to leverage config loader
    rag_engine = RAGEngine(session)
    try:
        rag_config = rag_engine.get_active_config(project_id)
    except Exception:
        from app.models.rag import RAGConfig
        rag_config = RAGConfig(
            project_id=project_id,
            chunk_size=500,
            chunk_overlap=50,
            max_tokens=4096,
            top_k=4,
            similarity_threshold=0.0,
            use_hybrid_search=True,
            semantic_weight=0.7,
            embedding_model="huggingface"
        )
    embeddings = rag_engine._get_embeddings(rag_config)
    
    # Load vector store from disk
    import os
    from langchain_community.vectorstores import FAISS
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
    else:
        vector_store = None
        
    inactive = rag_engine._inactive_doc_ids(project_id)
    
    candidate_k = max(rag_config.top_k * 5, 20)
    semantic_results = []
    bm25_results = []
    used_hybrid = False
    
    if vector_store:
        if strategy == "semantic":
            # Semantic search only
            results = vector_store.similarity_search_with_score(
                query, k=candidate_k, filter={"project_id": project_id}
            )
            for doc, score in results:
                did = doc.metadata.get("doc_id")
                if did is not None and int(did) in inactive:
                    continue
                semantic_results.append((doc, score))
                
        elif strategy == "hybrid":
            # Semantic + BM25 hybrid search
            results = vector_store.similarity_search_with_score(
                query, k=candidate_k, filter={"project_id": project_id}
            )
            for doc, score in results:
                did = doc.metadata.get("doc_id")
                if did is not None and int(did) in inactive:
                    continue
                semantic_results.append((doc, score))
                
            if bm25_manager.index_exists(str(project_id)):
                bm25_results = bm25_manager.search(str(project_id), query, top_k=candidate_k)
                used_hybrid = True
                
        elif strategy == "decomposed":
            # Search multiple sub-queries from query understanding stage
            seen_semantic = set()
            seen_bm25 = set()
            for sub_q in sub_queries:
                # Semantic subquery search
                sub_res = vector_store.similarity_search_with_score(
                    sub_q, k=candidate_k // 2, filter={"project_id": project_id}
                )
                for doc, score in sub_res:
                    did = doc.metadata.get("doc_id")
                    if did is not None and int(did) in inactive:
                        continue
                    if doc.page_content not in seen_semantic:
                        seen_semantic.add(doc.page_content)
                        semantic_results.append((doc, score))
                
                # BM25 subquery search (if hybrid configured or default)
                if bm25_manager.index_exists(str(project_id)):
                    sub_bm25 = bm25_manager.search(str(project_id), sub_q, top_k=candidate_k // 2)
                    for text, score in sub_bm25:
                        if text not in seen_bm25:
                            seen_bm25.add(text)
                            bm25_results.append((text, score))
            used_hybrid = True

    # RRF Hybrid Merge
    if used_hybrid and bm25_results:
        merged_results = hybrid_search_merge(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            semantic_weight=rag_config.semantic_weight,
            bm25_weight=1.0 - rag_config.semantic_weight
        )
    else:
        merged_results = semantic_results
        
    # fast TF-IDF Context Pruning
    prune_threshold = max(rag_config.similarity_threshold or 0.0, 0.1)
    pruned_results, orig_count, pruned_count, reduction_pct = context_pruner.prune(
        query=query,
        chunks=merged_results,
        threshold=prune_threshold
    )
    
    # BGE Cross-Encoder Reranking
    final_reranked = reranker_service.rerank(
        query=query,
        chunks=pruned_results,
        top_k=rag_config.top_k
    )
    
    # Prepare inputs for Source Confidence evaluation
    doc_ids = list(set(int(doc.metadata.get("doc_id")) for doc, _ in final_reranked if doc.metadata.get("doc_id") is not None))
    doc_meta_map = {}
    if doc_ids:
        from app.models.rag import Document as DBDocument
        docs = session.exec(select(DBDocument).where(DBDocument.id.in_(doc_ids))).all()
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
    
    # Format results for state
    formatted_results = []
    for doc, score in final_reranked:
        formatted_results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "doc_id": doc.metadata.get("doc_id")
        })
        
    context_str = "\n\n".join([r["content"] for r in formatted_results])
    
    strategies_tried = list(state.get("strategies_tried", []))
    if strategy not in strategies_tried:
        strategies_tried.append(strategy)
        
    latency_ms = (time.time() - start_time) * 1000.0
    
    return {
        "attempt_count": attempt,
        "strategies_tried": strategies_tried,
        "retrieval_results": formatted_results,
        "confidence_score": gate_result.max_confidence,
        "context": context_str,
        "_temp_latency": latency_ms,
        "_temp_gate_passed": gate_result.passed,
        "_temp_refusal_reason": gate_result.refusal_reason
    }


def evaluate_confidence(state: AgentState) -> dict:
    """Logs trace step details based on the evaluation outcome."""
    latency = state.get("_temp_latency", 0.0)
    confidence = state.get("confidence_score", 0.0)
    strategy = state["current_strategy"]
    attempt = state["attempt_count"]
    
    agent_trace = list(state.get("agent_trace", []))
    
    if confidence >= 0.75:
        action = "Generated response"
    elif attempt < 3:
        action = "Replanning..."
    else:
        action = "Cannot answer"
        
    step_trace = {
        "step": f"Attempt {attempt}",
        "strategy": strategy,
        "confidence": confidence,
        "action_taken": action,
        "latency_ms": latency
    }
    agent_trace.append(step_trace)
    
    return {
        "agent_trace": agent_trace
    }


def replan_retrieval(state: AgentState) -> dict:
    """Switches strategy to alternative modes or broadens the search query."""
    attempt = state["attempt_count"]
    current_strat = state["current_strategy"]
    
    next_strat = current_strat
    if attempt == 1:
        # Switch alternative retrieval mode
        if current_strat == "semantic":
            next_strat = "hybrid"
        else:
            next_strat = "semantic"
    elif attempt == 2:
        # Broaden query (switch to sub-queries)
        next_strat = "decomposed"
        
    return {
        "current_strategy": next_strat
    }


def generate_response(state: AgentState, config: dict = None) -> dict:
    """Passes context to the designated LLM to draft the final response."""
    configurable = (config or {}).get("configurable", {})
    session = configurable.get("session")
    if not session:
        raise ValueError("SQLModel Session is required.")
        
    query = state["query"]
    project_id = int(state["project_id"])
    context = state["context"]
    
    model_provider = configurable.get("model_provider", "groq")
    model_name = configurable.get("model_name", "llama-3.3-70b-versatile")
    temperature = configurable.get("temperature", 0.1)
    chat_history = configurable.get("chat_history", [])
    other_context_str = configurable.get("other_context_str", "")
    
    rag_engine = RAGEngine(session)
    try:
        rag_config = rag_engine.get_active_config(project_id)
        style = rag_config.response_style
    except Exception:
        style = "Concise"
        
    # Setup LLM based on user selection
    import os
    if model_provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model_name=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )
        
    system_prompt = f"""You are a helpful AI assistant.
    Current Project ID: {project_id}.
    Response Style: {style}.

    {other_context_str}

    IMPORTANT: Answer the question DIRECTLY based on the retrieved context below.
    Context:
    {context}
    
    Guidelines:
    1. Answer the question DIRECTLY based on the retrieved context.
    2. Do NOT start your answer with "The project contains..." or "The document mentions...". Just state the fact.
    3. Do NOT mention "The project ID is..." unless explicitly asked.
    """
    
    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=query))
    
    try:
        response_msg = llm.invoke(messages)
        response_text = response_msg.content
    except Exception as e:
        response_text = f"Error during response generation: {str(e)}"
        
    return {
        "response": response_text,
        "answered": True
    }


def cannot_answer(state: AgentState) -> dict:
    """Sets a structured refusal response."""
    response_content = (
        "I'm sorry, but I couldn't find a sufficiently confident answer in the available documents "
        "after trying multiple retrieval strategies."
    )
    return {
        "response": response_content,
        "answered": False
    }


# Define LangGraph StateGraph workflow
workflow = StateGraph(AgentState)

workflow.add_node("plan_retrieval", plan_retrieval)
workflow.add_node("execute_retrieval", execute_retrieval)
workflow.add_node("evaluate_confidence", evaluate_confidence)
workflow.add_node("replan_retrieval", replan_retrieval)
workflow.add_node("generate_response", generate_response)
workflow.add_node("cannot_answer", cannot_answer)

workflow.set_entry_point("plan_retrieval")

workflow.add_edge("plan_retrieval", "execute_retrieval")
workflow.add_edge("execute_retrieval", "evaluate_confidence")

def route_after_evaluation(state: AgentState):
    confidence = state.get("confidence_score", 0.0)
    attempt = state.get("attempt_count", 0)
    if confidence >= 0.75:
        return "generate"
    elif attempt < 3:
        return "replan"
    else:
        return "cannot_answer"

workflow.add_conditional_edges(
    "evaluate_confidence",
    route_after_evaluation,
    {
        "generate": "generate_response",
        "replan": "replan_retrieval",
        "cannot_answer": "cannot_answer"
    }
)

workflow.add_edge("replan_retrieval", "execute_retrieval")
workflow.add_edge("generate_response", END)
workflow.add_edge("cannot_answer", END)

# Compile the workflow
compiled_graph = workflow.compile()
