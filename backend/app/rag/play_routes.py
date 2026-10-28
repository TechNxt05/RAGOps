import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.auth.deps import get_current_user
from app.core.config import settings
from app.core.rate_limit import rate_limiter
from app.models.user import User
from app.services import llm_factory
from app.utils.llm_response import extract_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/playground", tags=["playground"])

class PlaygroundMessage(BaseModel):
    role: str
    content: str = Field(max_length=20000)

class PlaygroundRequest(BaseModel):
    messages: List[PlaygroundMessage] = Field(max_length=50)
    model_provider: str = "groq"
    model_name: str = "openai/gpt-oss-20b"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=1000, ge=1, le=8192)
    system_prompt: Optional[str] = Field(default=None, max_length=20000)

@router.post("/generate")
async def playground_generate(request: PlaygroundRequest, current_user: User = Depends(get_current_user)):
    rate_limiter.check(f"chat:{current_user.id}", settings.rate_limit_chat_per_min)
    provider = request.model_provider.lower()
    if provider not in llm_factory.SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {request.model_provider}")
    if not llm_factory.provider_available(provider):
        raise HTTPException(status_code=400, detail=f"No API key configured for provider '{provider}'")

    llm = llm_factory.build_chat_model(provider, request.model_name, request.temperature, request.max_tokens)

    messages = []
    if request.system_prompt and request.system_prompt.strip():
        messages.append(SystemMessage(content=request.system_prompt))
    for msg in request.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            messages.append(SystemMessage(content=msg.content))

    try:
        response = await llm.ainvoke(messages)
    except Exception as exc:
        logger.warning("Playground generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"The model provider returned an error: {str(exc)[:300]}")
    return {
        "content": extract_text(response.content),
        "usage": llm_factory.token_usage(response) or {},
    }
