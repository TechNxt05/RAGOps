"""
Single place that turns (provider, model) into a LangChain chat model.

Replaces the provider if/elif blocks that were copy-pasted across chat, engine,
playground, benchmark and agent code, and adds an ordered fallback chain so a
rate-limited free-tier model (HTTP 429) transparently rolls over to the next one.
"""

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("groq", "google", "openrouter", "openai", "anthropic")
REQUEST_TIMEOUT_S = 90

_PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
    "google": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def provider_available(provider: str) -> bool:
    env = _PROVIDER_KEYS.get(provider)
    return bool(env and os.getenv(env))


def infer_provider(model_name: str) -> str:
    lowered = model_name.lower()
    if "gemini" in lowered:
        return "google"
    if lowered.startswith("claude"):
        return "anthropic"
    if ":free" in lowered:
        return "openrouter"
    return "groq"


_model_cache: Dict[tuple, Any] = {}
_cache_lock = threading.Lock()


def build_chat_model(
    provider: str,
    name: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    streaming: bool = False,
    fast: bool = False,
):
    """Cached: building a client costs ~0.4 s (fresh HTTP/TLS setup), and chat
    model objects are stateless and safe to share across requests/threads.
    fast=True minimises hidden reasoning on reasoning models (for mechanical
    utility calls such as query rewriting, never for answer generation)."""
    provider = (provider or infer_provider(name)).lower()
    key = (provider, name, round(float(temperature), 3), max_tokens, streaming, fast)
    cached = _model_cache.get(key)
    if cached is not None:
        return cached
    with _cache_lock:
        cached = _model_cache.get(key)
        if cached is None:
            cached = _build_chat_model(provider, name, temperature, max_tokens, streaming, fast)
            if len(_model_cache) > 256:
                _model_cache.clear()
            _model_cache[key] = cached
    return cached


def _reasoning_kwargs(provider: str, name: str, fast: bool) -> Dict[str, Any]:
    if not fast or provider != "groq":
        return {}
    lowered = name.lower()
    if lowered.startswith("openai/gpt-oss"):
        return {"model_kwargs": {"reasoning_effort": "low"}}
    if lowered.startswith("qwen/qwen3"):
        return {"model_kwargs": {"reasoning_effort": "none"}}
    return {}


def _build_chat_model(provider: str, name: str, temperature: float, max_tokens: Optional[int], streaming: bool, fast: bool = False):
    common: Dict[str, Any] = {"temperature": temperature}
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=name, api_key=os.getenv("GROQ_API_KEY"), max_tokens=max_tokens,
            timeout=REQUEST_TIMEOUT_S, max_retries=1, streaming=streaming,
            **_reasoning_kwargs(provider, name, fast), **common,
        )
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=name, api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1",
            max_tokens=max_tokens, timeout=REQUEST_TIMEOUT_S, max_retries=1, streaming=streaming, **common,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=name, api_key=os.getenv("OPENAI_API_KEY"), max_tokens=max_tokens,
            timeout=REQUEST_TIMEOUT_S, max_retries=1, streaming=streaming, **common,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=name, api_key=os.getenv("ANTHROPIC_API_KEY"), max_tokens=max_tokens or 1024,
            timeout=REQUEST_TIMEOUT_S, max_retries=1, streaming=streaming, **common,
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=name, google_api_key=os.getenv("GEMINI_API_KEY"), max_output_tokens=max_tokens,
            timeout=REQUEST_TIMEOUT_S, max_retries=1, **common,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def model_chain(config=None, preferred: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """Ordered, de-duplicated [{provider, name}] list: the requested model first,
    then the project's primary and fallback models. Providers without an API
    key configured are skipped (unless nothing else is available)."""
    from app.models.rag import DEFAULT_FALLBACK_MODELS, DEFAULT_PRIMARY_MODELS
    candidates: List[Dict[str, str]] = []
    if preferred and preferred.get("name"):
        candidates.append({"provider": preferred.get("provider") or infer_provider(preferred["name"]), "name": preferred["name"]})
    if config is not None:
        candidates += list(getattr(config, "primary_models", None) or [])
        candidates += list(getattr(config, "fallback_models", None) or [])
    candidates += DEFAULT_PRIMARY_MODELS + DEFAULT_FALLBACK_MODELS

    seen = set()
    chain = []
    for m in candidates:
        provider = (m.get("provider") or infer_provider(m.get("name", ""))).lower()
        name = m.get("name")
        if not name or provider not in SUPPORTED_PROVIDERS or (provider, name) in seen:
            continue
        seen.add((provider, name))
        chain.append({"provider": provider, "name": name})
    available = [m for m in chain if provider_available(m["provider"])]
    return available or chain


def build_with_fallbacks(chain: List[Dict[str, str]], temperature: float = 0.2, max_tokens: Optional[int] = None, max_models: int = 4, fast: bool = False):
    """A runnable that tries each model in order on any error (429, timeout, 5xx,
    decommissioned model...). Returns (runnable, primary_spec)."""
    models = []
    for spec in chain[:max_models]:
        try:
            models.append(build_chat_model(spec["provider"], spec["name"], temperature, max_tokens, fast=fast))
        except Exception as exc:
            logger.warning("Skipping model %s/%s: %s", spec["provider"], spec["name"], exc)
    if not models:
        raise RuntimeError("No usable LLM configured. Set GROQ_API_KEY, GEMINI_API_KEY or OPENROUTER_API_KEY.")
    primary = models[0]
    return (primary.with_fallbacks(models[1:]) if len(models) > 1 else primary), chain[0]


def response_model_name(message: Any, default: str) -> str:
    meta = getattr(message, "response_metadata", None) or {}
    return meta.get("model_name") or meta.get("model") or default


def token_usage(message: Any) -> Optional[Dict[str, int]]:
    """Real provider-reported token counts when available."""
    usage = getattr(message, "usage_metadata", None)
    if usage:
        return {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
        }
    meta = getattr(message, "response_metadata", None) or {}
    tu = meta.get("token_usage") or meta.get("usage") or {}
    if tu:
        return {
            "input_tokens": int(tu.get("prompt_tokens", 0) or tu.get("input_tokens", 0) or 0),
            "output_tokens": int(tu.get("completion_tokens", 0) or tu.get("output_tokens", 0) or 0),
        }
    return None


# ---------------------------------------------------------------------------
# Live model catalog (so the UI never offers a decommissioned model)
# ---------------------------------------------------------------------------

_catalog_cache: Dict[str, Any] = {"at": 0.0, "data": None}
_STATIC_CATALOG = {
    "groq": ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
    "google": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.1-flash-lite"],
    "openrouter": [],
}
_EXCLUDE = ("whisper", "orpheus", "tts", "audio", "image", "guard", "embedding", "live", "omni")


def model_catalog(ttl_seconds: int = 3600) -> Dict[str, List[str]]:
    if _catalog_cache["data"] and time.time() - _catalog_cache["at"] < ttl_seconds:
        return _catalog_cache["data"]
    import httpx
    catalog: Dict[str, List[str]] = {}
    try:
        if provider_available("groq"):
            r = httpx.get("https://api.groq.com/openai/v1/models",
                          headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}, timeout=10)
            r.raise_for_status()
            catalog["groq"] = sorted(m["id"] for m in r.json().get("data", []) if not any(x in m["id"] for x in _EXCLUDE))
    except Exception as exc:
        logger.warning("Groq model list unavailable: %s", exc)
    try:
        if provider_available("google"):
            r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models",
                          params={"key": os.getenv("GEMINI_API_KEY"), "pageSize": 200}, timeout=10)
            r.raise_for_status()
            catalog["google"] = sorted(
                m["name"].split("/")[-1] for m in r.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
                and "gemini" in m["name"] and not any(x in m["name"] for x in _EXCLUDE)
            )
    except Exception as exc:
        logger.warning("Gemini model list unavailable: %s", exc)
    try:
        if provider_available("openrouter"):
            r = httpx.get("https://openrouter.ai/api/v1/models", timeout=10)
            r.raise_for_status()
            catalog["openrouter"] = sorted(m["id"] for m in r.json().get("data", []) if m["id"].endswith(":free"))
    except Exception as exc:
        logger.warning("OpenRouter model list unavailable: %s", exc)

    for provider, models in _STATIC_CATALOG.items():
        if provider_available(provider) and not catalog.get(provider):
            catalog[provider] = models
    _catalog_cache.update(at=time.time(), data=catalog)
    return catalog
