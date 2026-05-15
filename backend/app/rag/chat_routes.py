from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.auth.deps import get_current_admin, get_current_user
from app.db import get_session
from app.models.chat import ChatSession, Message
from app.models.query_log import QueryLog
from app.models.rag import RAGConfig
from app.models.usage import TokenUsage
from app.models.user import User
from app.mcp.registry import MCPRegistry
from app.mcp.schemas import MCPContext
from app.mcp.server import MCPServer
from app.rag.engine import RAGEngine
from app.services.rag_evaluator import evaluate_rag_response
from app.utils.cost import calculate_cost

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessageRequest(BaseModel):
    content: str
    project_id: Optional[int] = None
    session_id: Optional[int] = None
    temperature: float = 0.1
    model_provider: str = "groq"
    model_name: str = "llama-3.3-70b-versatile"
    history_limit: int = 5
    project_context_limit: int = 2
    context_session_ids: List[int] = Field(default_factory=list)
    title: Optional[str] = None


def _parse_tool_observation(obs: object) -> Tuple[List[str], List[dict]]:
    chunks: List[str] = []
    sources: List[dict] = []
    if obs is None:
        return chunks, sources
    if hasattr(obs, "content"):
        obs = getattr(obs, "content")
    if not isinstance(obs, str):
        obs = str(obs)
    try:
        data = json.loads(obs)
    except json.JSONDecodeError:
        return chunks, sources
    if isinstance(data, dict) and "chunks" in data and isinstance(data["chunks"], list):
        for ch in data["chunks"]:
            if isinstance(ch, dict):
                c = ch.get("content")
                if c:
                    chunks.append(str(c))
                src = ch.get("source")
                if src:
                    sources.append(
                        {
                            "source": str(src),
                            "doc_id": int(ch["doc_id"]) if ch.get("doc_id") is not None else 0,
                        }
                    )
    return chunks, sources


def _context_from_intermediate_steps(result: dict) -> Tuple[List[str], List[dict]]:
    chunks: List[str] = []
    sources: List[dict] = []
    for step in result.get("intermediate_steps") or []:
        if not isinstance(step, (tuple, list)) or len(step) < 2:
            continue
        action, observation = step[0], step[1]
        tool_name = getattr(action, "tool", None)
        if tool_name is None and hasattr(action, "tool_calls") and action.tool_calls:
            tc = action.tool_calls[0]
            tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if tool_name != "search_documents":
            continue
        c, s = _parse_tool_observation(observation)
        chunks.extend(c)
        sources.extend(s)
    return chunks, sources


@router.post("/message")
async def post_message(
    req: ChatMessageRequest,
    session_db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
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
    }
    session_db.add(chat_session)
    session_db.commit()

    user_msg = Message(session_id=session_id, role="user", content=content)
    session_db.add(user_msg)
    session_db.commit()

    past_messages = session_db.exec(
        select(Message)
        .where(Message.session_id == session_id)
        .where(Message.id != user_msg.id)
        .order_by(Message.created_at.desc())
        .limit(history_limit)
    ).all()
    past_messages = sorted(past_messages, key=lambda m: m.created_at)

    chat_history = []
    for msg in past_messages:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))

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

    mcp_context = MCPContext(
        user_id=current_user.id,
        user_role=current_user.role,
        project_id=project_id,
        session_id=session_id,
    )
    mcp_server = MCPServer(mcp_context)

    available_tools_meta = mcp_server.get_available_tools()
    langchain_tools = []

    for meta in available_tools_meta:
        tool_name = meta.name

        async def _tool_wrapper(
            tool_input: Optional[BaseModel] = None,
            tool_name: str = tool_name,
            **kwargs,
        ):
            if tool_input:
                kwargs = (
                    tool_input.model_dump()
                    if hasattr(tool_input, "model_dump")
                    else tool_input.dict()
                )
            output = await mcp_server.call_tool(tool_name, kwargs)
            return output.content

        tool_entry = MCPRegistry.get_tool(tool_name)
        input_model = tool_entry["model"]
        lc_tool = StructuredTool.from_function(
            func=None,
            coroutine=_tool_wrapper,
            name=tool_name,
            description=meta.description,
            args_schema=input_model,
        )
        langchain_tools.append(lc_tool)

    if model_provider == "groq":
        llm = ChatGroq(
            model_name=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    rag_engine = RAGEngine(session_db)
    try:
        rag_config = rag_engine.get_active_config(project_id)
        style = rag_config.response_style
    except Exception:
        rag_config = RAGConfig(project_id=project_id)
        style = "Concise"

    system_prompt = f"""You are a helpful AI assistant with access to tools.
    Current Project ID: {project_id}.
    Response Style: {style}.

    {other_context_str}

    IMPORTANT: You are an augmented AI.
    1. If the user asks a question, use 'search_documents' to find the answer.
    2. Answer the question DIRECTLY based on the retrieved content.
    3. Do NOT start your answer with "The project contains..." or "The document mentions...". Just state the fact.
    4. Do NOT mention "The project ID is..." unless explicitly asked.
    5. If asked for a summary, synthesize information from ALL retrieved chunks to provide a comprehensive overview.
    6. If the retrieved chunks contain a section explicitly labeled "SUMMARY" or "ABSTRACT", prioritize that information.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    try:
        agent = create_tool_calling_agent(llm, langchain_tools, prompt)
    except Exception as e:
        return {
            "session_id": session_id,
            "role": "assistant",
            "content": f"System Error: Failed to initialize AI Agent. Details: {str(e)}",
            "sources": [],
            "usage_metadata": None,
            "query_log_id": None,
            "quality": None,
        }

    agent_executor = AgentExecutor(
        agent=agent,
        tools=langchain_tools,
        verbose=True,
        return_intermediate_steps=True,
    )

    t0 = time.perf_counter()
    try:
        result = await agent_executor.ainvoke({"input": content, "chat_history": chat_history})
        answer = result["output"]
    except Exception as e:
        answer = f"Error during processing: {str(e)}"
        result = {"intermediate_steps": []}

    latency_ms = (time.perf_counter() - t0) * 1000.0

    ctx_chunks, src_from_tools = _context_from_intermediate_steps(result if isinstance(result, dict) else {})
    if not ctx_chunks and project_id is not None:
        try:
            cfg = rag_engine.get_active_config(project_id)
            raw = rag_engine.search(
                content,
                project_id,
                k=cfg.top_k,
                score_threshold=cfg.similarity_threshold,
            )
            ctx_chunks = [d.page_content for d, _ in raw]
            for d, _ in raw:
                meta = d.metadata or {}
                src_from_tools.append(
                    {
                        "source": str(meta.get("source", "Unknown")),
                        "doc_id": int(meta["doc_id"]) if meta.get("doc_id") is not None else 0,
                    }
                )
        except Exception:
            pass

    sync_scores = evaluate_rag_response(content, answer, ctx_chunks)

    usage_metadata = {
        "model": model_name,
        "provider": model_provider,
        "temperature": temperature,
        "history_limit": history_limit,
        "rag_config": {
            "chunk_size": rag_config.chunk_size,
            "chunk_overlap": rag_config.chunk_overlap,
            "similarity_threshold": getattr(rag_config, "similarity_threshold", 0.0),
            "max_tokens": rag_config.max_tokens or 4096,
            "response_style": rag_config.response_style,
        },
        "embeddings": "Google Gemini (embedding-001)",
        "context_used": [s.title for s in context_sessions] if context_sessions else "None",
        "timestamp": datetime.utcnow().isoformat(),
        "quality": {
            "hallucination_score": float(sync_scores["hallucination_score"]),
            "faithfulness_score": float(sync_scores["faithfulness_score"]),
            "overall_quality_score": float(sync_scores["overall_quality_score"]),
            "quality_label": str(sync_scores["quality_label"]),
        },
    }

    sources_out = src_from_tools if src_from_tools else []

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

    tokens_used = 0
    try:
        input_est = len(system_prompt) + len(content) + len(other_context_str)
        output_est = len(answer)
        i_tokens = int(input_est / 4)
        o_tokens = int(output_est / 4)
        tokens_used = i_tokens + o_tokens
        cost = calculate_cost(model_name, model_provider, i_tokens, o_tokens)
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
        print(f"Error tracking usage: {e}")

    model_used = f"{model_provider}/{model_name}"

    qlog = QueryLog(
        project_id=project_id,
        user_id=current_user.id,
        session_id=session_id,
        query_text=content,
        response_text=answer,
        model_used=model_used,
        latency_ms=latency_ms,
        chunks_retrieved=len(ctx_chunks),
        tokens_used=tokens_used,
        citations_shown=len(sources_out),
        citations_clicked=0,
        context_chunks_json=json.dumps(ctx_chunks[:50]),
        hallucination_score=float(sync_scores["hallucination_score"]),
        faithfulness_score=float(sync_scores["faithfulness_score"]),
    )
    session_db.add(qlog)
    session_db.commit()
    session_db.refresh(qlog)

    if is_new_session and not title:
        try:
            title_prompt = (
                f"Summarize this conversation into a very short 3-5 word title. "
                f"Do not use quotes. User: {content}\nAI: {answer}"
            )
            title_response = llm.invoke(title_prompt)
            new_title = title_response.content.strip()
            new_title = new_title.replace('"', "").replace("'", "").replace("**", "")[:50]
            chat_session.title = new_title
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
        "quality": {
            "hallucination_score": float(sync_scores["hallucination_score"]),
            "faithfulness_score": float(sync_scores["faithfulness_score"]),
            "overall_quality_score": float(sync_scores["overall_quality_score"]),
            "quality_label": str(sync_scores["quality_label"]),
        },
    }


class CompareModelsRequest(BaseModel):
    query: str
    project_id: int


@router.post("/compare-models")
async def compare_models(
    body: CompareModelsRequest,
    session_db: Session = Depends(get_session),
    current_user: User = Depends(get_current_admin),
):
    rag_engine = RAGEngine(session_db)
    config = rag_engine.get_active_config(body.project_id)
    raw = rag_engine.search(
        body.query,
        body.project_id,
        k=config.top_k,
        score_threshold=config.similarity_threshold,
    )
    ctx_chunks = [d.page_content for d, _ in raw]
    context_str = "\n\n".join(ctx_chunks[:24])
    base = (
        "You are a precise assistant. Answer using ONLY the context below. "
        "If the answer is not in the context, say you cannot find it.\n\n"
        f"Context:\n{context_str}\n\nQuestion: {body.query}\nAnswer:"
    )

    async def run_side(provider: str, name: str) -> dict:
        t0 = time.perf_counter()
        if provider == "groq":
            side_llm = ChatGroq(
                model_name=name,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=config.temperature,
            )
        else:
            side_llm = ChatGoogleGenerativeAI(
                model=name,
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=config.temperature,
            )
        msg = await side_llm.ainvoke(base)
        latency = (time.perf_counter() - t0) * 1000.0
        text = getattr(msg, "content", str(msg))
        scores = evaluate_rag_response(body.query, text, ctx_chunks)
        return {
            "provider": provider,
            "model": name,
            "content": text,
            "latency_ms": latency,
            "hallucination_score": scores["hallucination_score"],
            "faithfulness_score": scores["faithfulness_score"],
            "overall_quality_score": scores["overall_quality_score"],
            "quality_label": scores["quality_label"],
            "citations_count": len(ctx_chunks),
        }

    left, right = await asyncio.gather(
        run_side(config.primary_llm_provider, config.primary_llm_name),
        run_side(config.fallback_llm_provider, config.fallback_llm_name),
    )

    winner = "tie"
    if left["overall_quality_score"] > right["overall_quality_score"] + 0.02:
        winner = "primary"
    elif right["overall_quality_score"] > left["overall_quality_score"] + 0.02:
        winner = "fallback"
    elif left["latency_ms"] < right["latency_ms"] * 0.85:
        winner = "primary"
    elif right["latency_ms"] < left["latency_ms"] * 0.85:
        winner = "fallback"

    return {"left": left, "right": right, "winner": winner}


@router.get("/sessions")
def get_sessions(
    project_id: Optional[int] = None,
    session_db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    query = select(ChatSession).where(ChatSession.user_id == current_user.id)
    if project_id:
        query = query.where(ChatSession.project_id == project_id)
    sessions = session_db.exec(query.order_by(ChatSession.created_at.desc())).all()
    return sessions


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: int,
    session_db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    chat_session = session_db.get(ChatSession, session_id)
    if not chat_session or chat_session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    session_db.delete(chat_session)
    session_db.commit()
    return {"ok": True}


@router.get("/history/{session_id}")
def get_history(
    session_id: int,
    session_db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    chat_session = session_db.get(ChatSession, session_id)
    if not chat_session or chat_session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = session_db.exec(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    ).all()
    return messages
