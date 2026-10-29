"""
LiteLLM Gateway Service.
Provides a unified model gateway interface wrapping Groq, Gemini, Anthropic, and OpenRouter.
Implements automatic provider fallback, retry logic, rate-limit resilience, and credit shortage protection.
Ensures zero frontend crashes and uninterrupted user experience even if a provider hits quota limits.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

import app.utils.httpx_patch
from app.core import llm_keys
from app.utils.llm_response import extract_text

logger = logging.getLogger("LiteLLMGateway")


class GatewayResponse(BaseModel):
    content: str
    model_used: str
    provider_used: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    fallback_triggered: bool = False


# Canonical fallback chain when quotas or credit limits are encountered
DEFAULT_FALLBACK_CHAIN = [
    {"provider": "groq", "name": "qwen/qwen3.8-27b"},
    {"provider": "google", "name": "gemini-2.5-flash-lite"},
    {"provider": "groq", "name": "openai/gpt-oss-120b"},
    {"provider": "openrouter", "name": "openai/gpt-4o-mini"},
]


class LiteLLMGateway:
    """
    Unified LLM call gateway for multi-provider routing, fallback, and retry execution.
    """

    async def generate(
        self,
        prompt: str,
        models: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> GatewayResponse:
        """
        Executes generation call with automatic fallback retry logic across an ordered list of models.
        """
        call_chain = list(models) if models else []
        # Append resilient defaults if not already present
        for fb in DEFAULT_FALLBACK_CHAIN:
            if not any(m.get("provider") == fb["provider"] and m.get("name") == fb["name"] for m in call_chain):
                call_chain.append(fb)

        last_exc = None
        attempt_count = 0

        for model_cfg in call_chain:
            provider = model_cfg.get("provider", "groq")
            model_name = model_cfg.get("name", "qwen/qwen3.8-27b")
            attempt_count += 1

            # Check if required API key is configured
            if not llm_keys.get_key(provider.lower()):
                continue

            try:
                res_text = await self._call_provider(provider, model_name, prompt, temperature, max_tokens)
                if res_text and len(res_text.strip()) > 0:
                    return GatewayResponse(
                        content=res_text,
                        model_used=model_name,
                        provider_used=provider,
                        prompt_tokens=int(len(prompt) / 4),
                        completion_tokens=int(len(res_text) / 4),
                        fallback_triggered=(attempt_count > 1),
                    )
            except Exception as exc:
                err_msg = str(exc)
                logger.warning(
                    f"LiteLLMGateway: Provider '{provider}' model '{model_name}' failed "
                    f"({err_msg[:120]}...). Transparently trying next fallback in chain..."
                )
                last_exc = exc
                await asyncio.sleep(0.3)

        logger.error(f"LiteLLMGateway: All providers in fallback chain failed. Last error: {last_exc}")
        
        # Graceful fallback synthesizer: never return an unhandled error to the user
        fallback_answer = self._synthesize_fallback_response(prompt)
        return GatewayResponse(
            content=fallback_answer,
            model_used="deterministic-fallback",
            provider_used="system",
            fallback_triggered=True,
        )

    async def _call_provider(self, provider: str, model_name: str, prompt: str, temperature: float, max_tokens: int) -> str:
        prov = provider.lower()
        if prov == "groq":
            from langchain_groq import ChatGroq
            client = ChatGroq(
                model_name=model_name,
                api_key=llm_keys.get_key("groq"),
                temperature=temperature,
                max_tokens=min(max_tokens, 1024),
            )
            res = await client.ainvoke(prompt)
            return extract_text(res.content)
        elif prov == "openrouter":
            from langchain_openai import ChatOpenAI
            api_key = llm_keys.get_key("openrouter")
            client = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                openai_api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=min(max_tokens, 300),  # Keep low for credit safety
            )
            res = await client.ainvoke(prompt)
            return extract_text(res.content)
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            client = ChatGoogleGenerativeAI(
                model=model_name if "gemini" in model_name else "gemini-2.5-flash-lite",
                google_api_key=llm_keys.get_key("google"),
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            res = await client.ainvoke(prompt)
            return extract_text(res.content)

    def _synthesize_fallback_response(self, prompt: str) -> str:
        """
        Deterministic synthesis if all remote APIs are temporarily unreachable.
        Extracts relevant factual lines from prompt context to maintain UX.
        """
        if "Context:" in prompt:
            try:
                parts = prompt.split("Context:")
                context_part = parts[1].split("Question:")[0].strip()
                lines = [line.strip() for line in context_part.split("\n") if len(line.strip()) > 30]
                if lines:
                    return f"Based on the provided documents:\n\n{lines[0]}"
            except Exception:
                pass
        return "I am currently unable to reach the AI model providers to complete this request. Please try again in a moment."


litellm_gateway = LiteLLMGateway()
