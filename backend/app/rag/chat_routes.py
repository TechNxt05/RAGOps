from __future__ import annotations

import asyncio
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from starlette.concurrency import run_in_threadpool

from app.auth.access import require_project
from app.auth.deps import get_current_admin, get_current_user
from app.db import get_session
from app.models.chat import ChatSession, Message
from app.models.query_log import QueryLog
from app.models.user import User
from app.rag import chat_pipeline
from app.rag.engine import RAGEngine
from app.services import llm_factory
from app.services.rag_evaluator import evaluate_rag_response
from app.utils.llm_response import extract_text

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessageRequest(BaseModel):
    content: str = Field(max_length=8000)
    project_id: Optional[int] = None
    session_id: Optional[int] = None
    temperature: Optional[float] = Field(default=0.1, ge=0, le=2)
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    history_limit: int = Field(default=6, ge=0, le=20)
    project_context_limit: int = Field(default=0, ge=0, le=5)
    context_session_ids: List[int] = Field(default_factory=list)
    title: Optional[str] = None


@router.post("/message")
async def post_message(
    req: ChatMessageRequest,
    session_db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    return await chat_pipeline.run_turn(req, session_db, current_user)


@router.post("/message/stream")
async def post_message_stream(
    req: ChatMessageRequest,
    session_db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Same pipeline as /chat/message, streamed as Server-Sent Events."""
    return StreamingResponse(
        chat_pipeline.stream_turn(req, session_db, current_user),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class CompareModelsRequest(BaseModel):
    query: str
    project_id: int


@router.post("/compare-models")
async def compare_models(
    body: CompareModelsRequest,
    session_db: Session = Depends(get_session),
    current_user: User = Depends(get_current_admin),
):
    require_project(session_db, current_user, body.project_id)
    rag_engine = RAGEngine(session_db)
    config = rag_engine._config_or_default(body.project_id)
    raw = await run_in_threadpool(
        rag_engine.search, body.query, body.project_id, config.top_k, config.similarity_threshold,
    )
    ctx_chunks = [d.page_content for d, _ in raw]
    context_str = "\n\n".join(ctx_chunks[:24])
    base = (
        "You are a precise assistant. Answer using ONLY the context below. "
        "If the answer is not in the context, say you cannot find it.\n\n"
        f"Context:\n{context_str}\n\nQuestion: {body.query}\nAnswer:"
    )

    async def run_side(spec: dict) -> dict:
        t0 = time.perf_counter()
        try:
            llm = llm_factory.build_chat_model(spec["provider"], spec["name"], temperature=config.temperature)
            msg = await llm.ainvoke(base)
            text = extract_text(msg.content)
            error = None
        except Exception as exc:
            text, error = "", str(exc)[:300]
        latency = (time.perf_counter() - t0) * 1000.0
        scores = evaluate_rag_response(body.query, text, ctx_chunks) if text else {
            "hallucination_score": 1.0, "faithfulness_score": 0.0, "overall_quality_score": 0.0, "quality_label": "Error"}
        return {
            "provider": spec["provider"], "model": spec["name"], "content": text, "error": error,
            "latency_ms": latency,
            "hallucination_score": scores["hallucination_score"],
            "faithfulness_score": scores["faithfulness_score"],
            "overall_quality_score": scores["overall_quality_score"],
            "quality_label": scores["quality_label"],
            "citations_count": len(ctx_chunks),
        }

    chain = llm_factory.model_chain(config)
    primary = chain[0]
    fallback = next((m for m in chain[1:] if m != primary), primary)
    left, right = await asyncio.gather(run_side(primary), run_side(fallback))

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
    return session_db.exec(query.order_by(ChatSession.created_at.desc())).all()


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: int,
    session_db: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    chat_session = session_db.get(ChatSession, session_id)
    if not chat_session or chat_session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    # Analytics rows outlive the chat: detach them so the FK doesn't block the delete.
    for log in session_db.exec(select(QueryLog).where(QueryLog.session_id == session_id)).all():
        log.session_id = None
        session_db.add(log)
    for msg in session_db.exec(select(Message).where(Message.session_id == session_id)).all():
        session_db.delete(msg)
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
    return session_db.exec(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    ).all()
