"""
PolicyEngine — Resolves policy configurations based on named presets (FAST, STANDARD, DEEP, CRITICAL)
or custom YAML/dict policy definitions.
"""

from enum import Enum
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class PolicyPreset(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"
    CRITICAL = "critical"


class PolicyConfig(BaseModel):
    """
    Configurable runtime policy object controlling budget constraints,
    quality targets, and operational thresholds.
    """
    preset_name: str = "standard"
    quality_target: float = Field(default=0.85, ge=0.0, le=1.0)
    latency_budget_ms: int = Field(default=3000, gt=0)
    cost_budget: float = Field(default=0.02, ge=0.0)
    max_retries: int = Field(default=1, ge=0, le=3)
    max_graph_depth: int = Field(default=2, ge=1, le=5)
    verification_level: Literal["BASIC", "STANDARD", "STRICT"] = "STANDARD"
    allow_early_exit: bool = True
    allow_rerank_skip: bool = True


PRESET_POLICIES: Dict[PolicyPreset, PolicyConfig] = {
    PolicyPreset.FAST: PolicyConfig(
        preset_name="fast",
        quality_target=0.75,
        latency_budget_ms=1000,
        cost_budget=0.005,
        max_retries=0,
        max_graph_depth=1,
        verification_level="BASIC",
        allow_early_exit=True,
        allow_rerank_skip=True
    ),
    PolicyPreset.STANDARD: PolicyConfig(
        preset_name="standard",
        quality_target=0.85,
        latency_budget_ms=3000,
        cost_budget=0.02,
        max_retries=1,
        max_graph_depth=2,
        verification_level="STANDARD",
        allow_early_exit=True,
        allow_rerank_skip=True
    ),
    PolicyPreset.DEEP: PolicyConfig(
        preset_name="deep",
        quality_target=0.92,
        latency_budget_ms=6000,
        cost_budget=0.05,
        max_retries=2,
        max_graph_depth=3,
        verification_level="STRICT",
        allow_early_exit=False,
        allow_rerank_skip=False
    ),
    PolicyPreset.CRITICAL: PolicyConfig(
        preset_name="critical",
        quality_target=0.98,
        latency_budget_ms=10000,
        cost_budget=0.10,
        max_retries=3,
        max_graph_depth=4,
        verification_level="STRICT",
        allow_early_exit=False,
        allow_rerank_skip=False
    )
}


class PolicyEngine:
    """
    Evaluates policy configuration for incoming state and query requirements.
    """

    def resolve_policy(
        self,
        preset: Optional[PolicyPreset] = None,
        custom_override: Optional[Dict[str, Any]] = None
    ) -> PolicyConfig:
        """
        Returns a resolved PolicyConfig for a preset, applying optional custom overrides.
        """
        base = PRESET_POLICIES.get(preset or PolicyPreset.STANDARD, PRESET_POLICIES[PolicyPreset.STANDARD])
        if not custom_override:
            return base

        merged = base.model_dump()
        merged.update(custom_override)
        return PolicyConfig(**merged)
