"""
Stateless question answering over a project, for API-key / MCP callers that
have no chat session. Uses the same retrieval, confidence gate, prompt and
model fallback chain as the chat pipeline.
"""

import time
from types import SimpleNamespace
from typing import Optional

from sqlmodel import Session
from starlette.concurrency import run_in_threadpool

from app.rag import chat_pipeline
from app.rag.engine import RAGEngine
from app.services import llm_factory
from app.utils.llm_response import extract_text


async def answer_question(db: Session, project_id: int, question: str, model_name: Optional[str] = None) -> dict:
    t0 = time.time()
    engine = RAGEngine(db)
    config = engine._config_or_default(project_id)
    preferred = {"provider": llm_factory.infer_provider(model_name), "name": model_name} if model_name else None
    chain = llm_factory.model_chain(config, preferred)
    utility_llm, _ = llm_factory.build_with_fallbacks(chain, temperature=0.0, max_tokens=2048, fast=True)

    results = await run_in_threadpool(engine.search, question, project_id, config.top_k,
                                      config.similarity_threshold, None, None, utility_llm)
    gate = getattr(results, "confidence_gate_result", None)
    turn = chat_pipeline.Turn(req=SimpleNamespace(content=question, temperature=None), user=None, db=db)
    turn.project_id, turn.config, turn.engine, turn.chain = project_id, config, engine, chain
    turn.results = [(d, float(s)) for d, s in results]
    turn.conflict_res = getattr(results, "conflict_detection", None)
    turn.query_analysis = getattr(results, "query_analysis", None)

    if gate is not None and not gate.passed:
        return {
            "answer": gate.refusal_reason,
            "refused": True,
            "confidence": gate.max_confidence,
            "sources": chat_pipeline.sources_of(turn.results),
            "latency_ms": round((time.time() - t0) * 1000.0, 1),
        }

    messages, contract, _ = chat_pipeline.build_messages(turn)
    llm = chat_pipeline.generation_llm(turn, chat_pipeline.generation_budget(turn, contract))
    result = await llm.ainvoke(messages)
    return {
        "answer": extract_text(result.content),
        "refused": False,
        "confidence": gate.max_confidence if gate else None,
        "model": llm_factory.response_model_name(result, chain[0]["name"]),
        "sources": chat_pipeline.sources_of(turn.results),
        "latency_ms": round((time.time() - t0) * 1000.0, 1),
    }
