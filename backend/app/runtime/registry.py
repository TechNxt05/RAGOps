"""
ModelRegistry — Capability profiles for models and retrieval backends.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ModelProfile(BaseModel):
    model_name: str
    provider: str
    context_window: int = 128000
    reasoning_level: str = "STANDARD"  # LOW, STANDARD, HIGH
    estimated_cost_per_1k: float = 0.001
    expected_latency_ms: int = 800
    supports_tools: bool = True
    supports_json: bool = True


DEFAULT_MODEL_PROFILES: Dict[str, ModelProfile] = {
    "groq/qwen/qwen3.8-27b": ModelProfile(
        model_name="groq/qwen/qwen3.8-27b",
        provider="groq",
        context_window=128000,
        reasoning_level="STANDARD",
        estimated_cost_per_1k=0.0007,
        expected_latency_ms=600,
        supports_tools=True,
        supports_json=True
    ),
    "gemini/gemini-3.7-flash": ModelProfile(
        model_name="gemini/gemini-3.7-flash",
        provider="gemini",
        context_window=1000000,
        reasoning_level="STANDARD",
        estimated_cost_per_1k=0.0003,
        expected_latency_ms=450,
        supports_tools=True,
        supports_json=True
    ),
    "gemini/gemini-3.1-flash-lite": ModelProfile(
        model_name="gemini/gemini-3.1-flash-lite",
        provider="gemini",
        context_window=1000000,
        reasoning_level="LOW",
        estimated_cost_per_1k=0.0001,
        expected_latency_ms=300,
        supports_tools=True,
        supports_json=True
    ),
    "anthropic/claude-3-5-sonnet": ModelProfile(
        model_name="anthropic/claude-3-5-sonnet",
        provider="anthropic",
        context_window=200000,
        reasoning_level="HIGH",
        estimated_cost_per_1k=0.0030,
        expected_latency_ms=1200,
        supports_tools=True,
        supports_json=True
    )
}


class ModelRegistry:
    """Registry maintaining active model capability profiles."""

    def __init__(self):
        self._profiles: Dict[str, ModelProfile] = dict(DEFAULT_MODEL_PROFILES)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all active registered models."""
        return [
            {"name": k, "provider": v.provider, "reasoning_level": v.reasoning_level, "context_window": v.context_window}
            for k, v in self._profiles.items()
        ]

    def get_profile(self, model_name: str) -> ModelProfile:
        return self._profiles.get(model_name, DEFAULT_MODEL_PROFILES["gemini/gemini-3.7-flash"])

    def select_best_model(self, max_latency_ms: int, max_cost: float, reasoning_level: str) -> str:
        """Selects optimal model candidate satisfying constraints."""
        candidates = []
        for name, profile in self._profiles.items():
            if profile.expected_latency_ms <= max_latency_ms:
                candidates.append(profile)

        if not candidates:
            return "gemini/gemini-3.7-flash"  # Fastest fallback

        if reasoning_level == "HIGH":
            high_candidates = [c for c in candidates if c.reasoning_level == "HIGH"]
            if high_candidates:
                return high_candidates[0].model_name

        # Return default fast versatile model
        return "gemini/gemini-3.7-flash"
