from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.auth.deps import get_current_user
from app.db import get_session
from app.models.chat import ChatSession, Message
from app.models.query_log import QueryLog
from app.models.usage import TokenUsage
from app.models.user import User
from app.models.rag import RAGConfig
from app.services.cost_control import get_cost_manager
from app.utils.cost import calculate_cost
from app.services.rag_evaluator import evaluate_rag_response

# Import existing router logic to keep standard mode identical
from app.rag.chat_routes import post_message, ChatMessageRequest
from app.agents import retrieval_agent

router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("")
async def post_query(
    req: ChatMessageRequest,
    session_db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Standard Query endpoint proxying to the existing /chat/message logic."""
    return await post_message(req, session_db, current_user)


@router.post("/agentic")
async def post_agentic_query(
    req: ChatMessageRequest,
    session_db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Agentic Query endpoint executing the autonomous LangGraph retrieval loop."""
    t0_overall = time.time()
    content = req.content
    project_id = req.project_id
    session_id = req.session_id
    temperature = req.temperature
    model_provider = req.model_provider
    model_name = req.model_name
    history_limit = req.history_limit
    project_context_limit = req.project_context_limit
    context_session_ids = req.context_session_ids or []
    title = req.title

    if session_id:
        chat_session = session_db.get(ChatSession, session_id)
        if not chat_session or chat_session.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Session not found")
        project_id = chat_session.project_id
        is_new_session = False
    else:
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id is required for new session")
        chat_session = ChatSession(
            user_id=current_user.id,
            title=title or content[:30],
            project_id=project_id,
        )
        session_db.add(chat_session)
        session_db.commit()
        session_db.refresh(chat_session)
        session_id = chat_session.id
        is_new_session = True

    chat_session.settings = {
        "model_provider": model_provider,
        "model_name": model_name,
        "temperature": temperature,
        "history_limit": history_limit,
        "project_context_limit": project_context_limit,
        "agentic": True
    }
    session_db.add(chat_session)
    session_db.commit()

    user_msg = Message(session_id=session_id, role="user", content=content)
    session_db.add(user_msg)
    session_db.commit()

    # 1. Cost Control Pre-Call
    cost_manager = get_cost_manager()
    pre = cost_manager.pre_call(content)

    if pre["source"] == "blocked":
        raise HTTPException(status_code=503, detail="Cost circuit breaker tripped. Hourly/daily spending limit reached.")

    # Route model dynamically if routed by query router
    routed_model = pre.get("model", model_name)
    routed_provider = "google" if "gemini" in routed_model.lower() else "groq"
    if pre.get("degraded") or routed_model != model_name:
        model_name = routed_model
        model_provider = routed_provider

    # Get chat history
    past_messages = session_db.exec(
        select(Message)
        .where(Message.session_id == session_id)
        .where(Message.id != user_msg.id)
        .order_by(Message.created_at.desc())
        .limit(history_limit)
    ).all()
    past_messages = sorted(past_messages, key=lambda m: m.created_at)

    from langchain_core.messages import AIMessage, HumanMessage
    chat_history = []
    for msg in past_messages:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))

    # Fetch context from related chats
    other_context_str = ""
    context_sessions = []
    if context_session_ids:
        context_sessions = session_db.exec(
            select(ChatSession)
            .where(ChatSession.id.in_(context_session_ids))
            .where(ChatSession.user_id == current_user.id)
            .where(ChatSession.id != session_id)
        ).all()
    elif project_context_limit > 0:
        context_sessions = session_db.exec(
            select(ChatSession)
            .where(ChatSession.project_id == project_id)
            .where(ChatSession.id != session_id)
            .order_by(ChatSession.created_at.desc())
            .limit(project_context_limit)
        ).all()

    if context_sessions:
        other_context_str = "\n\n### RELATED PROJECT CHATS (CONTEXT):\n"
        for osess in context_sessions:
            osess_msgs = session_db.exec(
                select(Message)
                .where(Message.session_id == osess.id)
                .order_by(Message.created_at.desc())
                .limit(3)
            ).all()
            osess_msgs = sorted(osess_msgs, key=lambda m: m.created_at)
            if osess_msgs:
                other_context_str += f"- Chat '{osess.title}':\n"
                for m in osess_msgs:
                    other_context_str += f"  {m.role.upper()}: {m.content[:200]}...\n"

    # Prepare inputs for LangGraph Agent
    initial_state = {
        "query": content,
        "project_id": str(project_id),
        "attempt_count": 0,
        "strategies_tried": [],
        "agent_trace": [],
        "answered": False
    }

    # Pass session and LLM params in configuration
    config = {
        "configurable": {
            "session": session_db,
            "model_provider": model_provider,
            "model_name": model_name,
            "temperature": temperature,
            "chat_history": chat_history,
            "other_context_str": other_context_str
        }
    }

    # Execute LangGraph retrieval agent
    try:
        final_state = await retrieval_agent.ainvoke(initial_state, config)
        answer = final_state.get("response", "Could not generate a response.")
        answered = final_state.get("answered", False)
        agent_trace = final_state.get("agent_trace", [])
        strategies_tried = final_state.get("strategies_tried", [])
        attempts = final_state.get("attempt_count", 0)
        retrieved_results = final_state.get("retrieval_results", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent runtime failure: {str(e)}")

    latency_ms = (time.time() - t0_overall) * 1000.0

    # Package pipeline trace details for storage
    pipeline_trace = {
        "query": content,
        "status": "success" if answered else "failed",
        "total_duration_ms": latency_ms,
        "agentic": True,
        "answered": answered,
        "attempts": attempts,
        "strategies_tried": strategies_tried,
        "agent_trace": agent_trace
    }

    # Evaluate answer quality (if answer is generated)
    ctx_chunks = [r["content"] for r in retrieved_results]
    if answered:
        sync_scores = evaluate_rag_response(content, answer, ctx_chunks)
        quality = {
            "hallucination_score": float(sync_scores["hallucination_score"]),
            "faithfulness_score": float(sync_scores["faithfulness_score"]),
            "overall_quality_score": float(sync_scores["overall_quality_score"]),
            "quality_label": str(sync_scores["quality_label"]),
        }
    else:
        quality = {
            "hallucination_score": 0.0,
            "faithfulness_score": 0.0,
            "overall_quality_score": 0.0,
            "quality_label": "Refused (Low Confidence)"
        }

    # Create usage metadata payload for message turn
    usage_metadata = {
        "model": model_name,
        "provider": model_provider,
        "temperature": temperature,
        "history_limit": history_limit,
        "timestamp": datetime.utcnow().isoformat(),
        "quality": quality,
        "agentic": True,
        "agent_trace": agent_trace,
        "attempts": attempts,
        "strategies_tried": strategies_tried,
        "answered": answered
    }

    # Format sources for UI
    sources_out = []
    for r in retrieved_results:
        sources_out.append({
            "source": str(r["source"]),
            "doc_id": int(r["doc_id"]) if r.get("doc_id") is not None else 0
        })

    assistant_msg = Message(
        session_id=session_id,
        role="assistant",
        content=answer,
        sources=json.dumps(sources_out) if sources_out else "[]",
        usage_metadata=usage_metadata,
    )
    session_db.add(assistant_msg)
    session_db.commit()
    session_db.refresh(assistant_msg)

    # Cost Tracking
    tokens_used = 0
    try:
        input_est = len(content) + len(other_context_str) + sum(len(c) for c in ctx_chunks)
        output_est = len(answer)
        i_tokens = int(input_est / 4)
        o_tokens = int(output_est / 4)
        tokens_used = i_tokens + o_tokens
        cost = calculate_cost(model_name, model_provider, i_tokens, o_tokens)
        
        cost_manager.post_call(content, answer, cost, pre.get("tier", "standard"))
        
        usage_record = TokenUsage(
            project_id=project_id,
            user_id=current_user.id,
            session_id=session_id,
            model=model_name,
            provider=model_provider,
            input_tokens=i_tokens,
            output_tokens=o_tokens,
            total_tokens=tokens_used,
            cost=cost,
        )
        session_db.add(usage_record)
        session_db.commit()
    except Exception as e:
        print(f"Error tracking agentic query usage: {e}")

    # Log Query Telemetry
    qlog = QueryLog(
        project_id=project_id,
        user_id=current_user.id,
        session_id=session_id,
        query_text=content,
        response_text=answer,
        model_used=f"{model_provider}/{model_name} (Agentic)",
        latency_ms=latency_ms,
        chunks_retrieved=len(ctx_chunks),
        tokens_used=tokens_used,
        citations_shown=len(sources_out),
        citations_clicked=0,
        context_chunks_json=json.dumps(ctx_chunks[:50]),
        hallucination_score=quality["hallucination_score"],
        faithfulness_score=quality["faithfulness_score"],
        chunks_before_pruning=len(ctx_chunks),
        chunks_after_pruning=len(ctx_chunks),
        pruning_reduction_pct=0.0,
        used_hybrid_search=any(s == "hybrid" or s == "decomposed" for s in strategies_tried),
        pipeline_trace=pipeline_trace
    )
    session_db.add(qlog)
    session_db.commit()
    session_db.refresh(qlog)

    # Titling for new session
    if is_new_session and not title:
        try:
            # Quick summary fallback title
            chat_session.title = content[:30]
            session_db.add(chat_session)
            session_db.commit()
        except Exception as e:
            print(f"DEBUG: Failed to auto-title session: {e}")

    return {
        "session_id": session_id,
        "role": "assistant",
        "content": answer,
        "sources": sources_out,
        "usage_metadata": usage_metadata,
        "query_log_id": qlog.id,
        "quality": quality,
        "agent_trace": agent_trace,
        "attempts": attempts,
        "strategies_tried": strategies_tried,
        "answered": answered
    }
