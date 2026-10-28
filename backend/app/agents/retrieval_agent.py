import time
from typing import List, TypedDict, Optional
from sqlmodel import select

from langgraph.graph import StateGraph, END

# Import existing services & helpers
from app.services.rrf_service import hybrid_search_merge
from app.services.reranker_service import reranker_service
from app.services.query_understanding import get_query_understanding
from app.rag.engine import RAGEngine

class AgentState(TypedDict, total=False):
    query: str
    confidence_threshold: float          # project's calibrated gate threshold
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
        )
    candidate_k = max(rag_config.top_k * 5, 20)
    semantic_results = []
    bm25_results = []
    used_hybrid = False

    if strategy == "decomposed":
        seen = set()
        for sub_q in sub_queries:
            for doc, score in rag_engine._single_hybrid_search(sub_q, project_id, max(candidate_k // 2, 10)):
                key = doc.metadata.get("chunk_id") or doc.page_content
                if key not in seen:
                    seen.add(key)
                    semantic_results.append((doc, score))
        used_hybrid = True
    else:
        semantic_results = rag_engine.semantic_search(query, project_id, candidate_k)
        if strategy == "hybrid":
            bm25_results = rag_engine.keyword_search(query, project_id, candidate_k)
            used_hybrid = True

    if used_hybrid and bm25_results:
        merged_results = hybrid_search_merge(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            semantic_weight=rag_config.semantic_weight,
            bm25_weight=1.0 - rag_config.semantic_weight
        )
    else:
        merged_results = semantic_results

    orig_count = len(merged_results)
    final_reranked, scored = reranker_service.rerank_scored(query, merged_results, rag_config.top_k)
    relevance = rag_engine.relevance_scores(query, final_reranked, rag_config, scored)
    final_reranked = [(doc, rel) for (doc, _), rel in zip(final_reranked, relevance)]
    pruned_count = len(final_reranked)
    reduction_pct = round(100.0 * (1 - pruned_count / orig_count), 2) if orig_count else 0.0
    gate_result = rag_engine.evaluate_confidence(final_reranked, rag_config)

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
        "_temp_refusal_reason": gate_result.refusal_reason,
        "confidence_threshold": float(getattr(rag_config, "confidence_threshold", 0.4) or 0.4),
    }


def evaluate_confidence(state: AgentState) -> dict:
    """Logs trace step details based on the evaluation outcome."""
    latency = state.get("_temp_latency", 0.0)
    confidence = state.get("confidence_score", 0.0)
    strategy = state["current_strategy"]
    attempt = state["attempt_count"]
    
    agent_trace = list(state.get("agent_trace", []))
    
    if confidence >= state.get("confidence_threshold", 0.4):
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
    model_name = configurable.get("model_name")
    temperature = configurable.get("temperature", 0.1)
    chat_history = configurable.get("chat_history", [])
    other_context_str = configurable.get("other_context_str", "")
    
    rag_engine = RAGEngine(session)
    try:
        rag_config = rag_engine.get_active_config(project_id)
        style = rag_config.response_style
    except Exception:
        style = "Concise"
        
    from app.services.llm_factory import build_with_fallbacks, model_chain
    try:
        cfg = rag_engine.get_active_config(project_id)
    except Exception:
        cfg = None
    preferred = {"provider": model_provider, "name": model_name} if model_name else None
    llm, _ = build_with_fallbacks(model_chain(cfg, preferred), temperature=temperature)

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
        from app.utils.llm_response import extract_text
        response_msg = llm.invoke(messages)
        response_text = extract_text(response_msg.content)
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
    if confidence >= state.get("confidence_threshold", 0.4):
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
