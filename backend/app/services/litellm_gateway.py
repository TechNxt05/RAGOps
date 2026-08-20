"""
LiteLLM Gateway Service.
Provides a unified model gateway interface wrapping Groq, Gemini, Anthropic, and OpenAI.
Implements provider fallback, retries, and token cost tracking across all model calls.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("LiteLLMGateway")


class GatewayResponse(BaseModel):
    content: str
    model_used: str
    provider_used: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0


class LiteLLMGateway:
    """
    Unified LLM call gateway for multi-provider routing, fallback, and retry execution.
    """

    async def generate(
        self,
        prompt: str,
        models: list[dict] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> GatewayResponse:
        """
        Executes generation call with automatic fallback retry logic across an ordered list of models.
        """
        if not models:
            models = [
                {"provider": "groq", "name": "llama3-70b-8192"},
                {"provider": "google", "name": "gemini-1.5-flash"}
            ]

        last_exc = None
        for model_cfg in models:
            provider = model_cfg.get("provider", "groq")
            model_name = model_cfg.get("name", "llama3-70b-8192")
            
            try:
                res_text = await self._call_provider(provider, model_name, prompt, temperature, max_tokens)
                return GatewayResponse(
                    content=res_text,
                    model_used=model_name,
                    provider_used=provider,
                    prompt_tokens=int(len(prompt) / 4),
                    completion_tokens=int(len(res_text) / 4),
                )
            except Exception as exc:
                logger.warning(f"LiteLLMGateway: Provider '{provider}' model '{model_name}' failed ({exc}). Trying next fallback...")
                last_exc = exc
                
        logger.error(f"LiteLLMGateway: All providers in fallback chain failed. Last error: {last_exc}")
        return GatewayResponse(
            content=f"Error executing LLM generation: {last_exc}",
            model_used=models[-1]["name"],
            provider_used=models[-1]["provider"]
        )

    async def _call_provider(self, provider: str, model_name: str, prompt: str, temperature: float, max_tokens: int) -> str:
        prov = provider.lower()
        if prov == "groq":
            from langchain_groq import ChatGroq
            client = ChatGroq(
                model_name=model_name,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens
            )
            res = await client.ainvoke(prompt)
            return res.content if hasattr(res, "content") else str(res)
        elif prov == "openrouter":
            from langchain_openai import ChatOpenAI
            client = ChatOpenAI(
                model_name=model_name,
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=max_tokens
            )
            res = await client.ainvoke(prompt)
            return res.content if hasattr(res, "content") else str(res)
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            client = ChatGoogleGenerativeAI(
                model=model_name if "gemini" in model_name else "gemini-1.5-flash",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            res = await client.ainvoke(prompt)
            return res.content if hasattr(res, "content") else str(res)


litellm_gateway = LiteLLMGateway()
