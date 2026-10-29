"""
AutoRAGBench — Pre-configured Model Profiles & Presets

Defines standard model configurations across OpenRouter, Google Gemini, and Groq.
Allows 1-click selection in AutoRAGBench and throughout the RAGOps platform without
requiring manual script or environment modifications.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.core import llm_keys


class ModelProfile(BaseModel):
    id: str
    name: str
    provider: str  # "openrouter" | "google" | "groq"
    model_name: str  # API identifier passed to provider
    description: str
    context_window: int = 128000
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    is_available: bool = True
    badge: str = "Standard"
    recommended_for_benchmark: bool = False


# ------------------------------------------------------------------
# Canonical Model Presets
# ------------------------------------------------------------------

MODEL_PROFILES: Dict[str, ModelProfile] = {
    "openrouter_gpt4o_mini": ModelProfile(
        id="openrouter_gpt4o_mini",
        name="GPT-4o Mini (OpenRouter)",
        provider="openrouter",
        model_name="openai/gpt-4o-mini",
        description="Fast, cost-effective OpenAI model with high instruction adherence and stability.",
        context_window=128000,
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        badge="Recommended",
        recommended_for_benchmark=True,
    ),
    "openrouter_qwen72b": ModelProfile(
        id="openrouter_qwen72b",
        name="Qwen 2.5 72B Instruct (OpenRouter)",
        provider="openrouter",
        model_name="qwen/qwen-2.5-72b-instruct",
        description="Top-tier open weights reasoning model with state-of-the-art synthetic benchmark performance.",
        context_window=131072,
        cost_per_1m_input=0.35,
        cost_per_1m_output=0.40,
        badge="High Reasoning",
        recommended_for_benchmark=True,
    ),
    "openrouter_llama70b": ModelProfile(
        id="openrouter_llama70b",
        name="LLaMA 3.3 70B Instruct (OpenRouter)",
        provider="openrouter",
        model_name="meta-llama/llama-3.3-70b-instruct",
        description="Meta's flagship open model with strong reasoning, tool-use, and factual recall capabilities.",
        context_window=131072,
        cost_per_1m_input=0.12,
        cost_per_1m_output=0.30,
        badge="Flagship Open",
        recommended_for_benchmark=True,
    ),
    "google_gemini_flash": ModelProfile(
        id="google_gemini_flash",
        name="Gemini 2.5 Flash Lite (Google Direct)",
        provider="google",
        model_name="gemini-2.5-flash-lite",
        description="High throughput, ultra-large 1M token context model via Google Generative AI API.",
        context_window=1048576,
        cost_per_1m_input=0.075,
        cost_per_1m_output=0.30,
        badge="Large Context",
        recommended_for_benchmark=True,
    ),
    "groq_qwen27b": ModelProfile(
        id="groq_qwen27b",
        name="Qwen 3.8 27B (Groq LPUs)",
        provider="groq",
        model_name="qwen/qwen3.8-27b",
        description="Ultra-low latency inference powered by Groq LPU silicon. Zero credit rate limits.",
        context_window=128000,
        cost_per_1m_input=0.00,
        cost_per_1m_output=0.00,
        badge="Ultra-Fast (Free)",
        recommended_for_benchmark=True,
    ),
    "groq_gpt120b": ModelProfile(
        id="groq_gpt120b",
        name="GPT-OSS 120B (Groq LPUs)",
        provider="groq",
        model_name="openai/gpt-oss-120b",
        description="Massive 120B parameter open-weights model running on Groq LPUs at hundreds of tokens/sec.",
        context_window=128000,
        cost_per_1m_input=0.00,
        cost_per_1m_output=0.00,
        badge="Massive 120B",
        recommended_for_benchmark=True,
    ),
    "groq_gpt20b": ModelProfile(
        id="groq_gpt20b",
        name="GPT-OSS 20B (Groq LPUs)",
        provider="groq",
        model_name="openai/gpt-oss-20b",
        description="High-speed 20B model for real-time extraction and factoid summarization on Groq.",
        context_window=128000,
        cost_per_1m_input=0.00,
        cost_per_1m_output=0.00,
        badge="High Speed",
        recommended_for_benchmark=False,
    ),
    "groq_qwen_reasoning": ModelProfile(
        id="groq_qwen_reasoning",
        name="Qwen 3.6 27B Reasoning (Groq)",
        provider="groq",
        model_name="qwen/qwen3.6-27b",
        description="Specialized reasoning mode model with step-by-step thinking traces on Groq.",
        context_window=128000,
        cost_per_1m_input=0.00,
        cost_per_1m_output=0.00,
        badge="Reasoning",
        recommended_for_benchmark=False,
    ),
    "openrouter_deepseek_r1": ModelProfile(
        id="openrouter_deepseek_r1",
        name="DeepSeek R1 (OpenRouter)",
        provider="openrouter",
        model_name="deepseek/deepseek-r1",
        description="DeepSeek's frontier reasoning model featuring step-by-step chain-of-thought verification.",
        context_window=64000,
        cost_per_1m_input=0.55,
        cost_per_1m_output=2.19,
        badge="Frontier Reasoning",
        recommended_for_benchmark=False,
    ),
}


def get_available_profiles() -> List[Dict]:
    """
    Returns list of model profiles annotated with live API key availability.
    """
    has_openrouter = bool(llm_keys.get_key("openrouter"))
    has_gemini = bool(llm_keys.get_key("google"))
    has_groq = bool(llm_keys.get_key("groq"))

    results = []
    for profile in MODEL_PROFILES.values():
        data = profile.dict()
        if profile.provider == "openrouter":
            data["is_available"] = has_openrouter
        elif profile.provider == "google":
            data["is_available"] = has_gemini
        elif profile.provider == "groq":
            data["is_available"] = has_groq
        results.append(data)
    return results


def get_profile_by_id(profile_id: str) -> Optional[ModelProfile]:
    """Retrieve a specific profile by identifier."""
    return MODEL_PROFILES.get(profile_id)
