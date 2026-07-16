"""
AutoRAGBench — Pipeline Variant Definitions

Defines the 6 canonical RAG pipeline variants benchmarked by the suite.
Each variant is a configuration dict consumed by the runner to selectively
enable / disable pipeline stages.
"""

from enum import Enum
from typing import Dict, Any


class PipelineVariant(str, Enum):
    NAIVE = "naive"
    ADVANCED = "advanced"
    HYBRID = "hybrid"
    CORRECTIVE = "corrective"
    AGENTIC = "agentic"
    FULL_RAGOPS = "full_ragops"


# ------------------------------------------------------------------
# Variant configuration maps
# ------------------------------------------------------------------
# Each key maps to a toggle or override understood by the benchmark
# runner when it constructs a synthetic RAGConfig for the run.
# ------------------------------------------------------------------

VARIANT_CONFIGS: Dict[PipelineVariant, Dict[str, Any]] = {
    PipelineVariant.NAIVE: {
        "label": "Naive RAG",
        "description": "Baseline semantic retrieval only (no reranker, no BM25, no query rewriting).",
        "use_hybrid_search": False,
        "use_reranker": False,
        "use_query_rewriting": False,
        "use_multi_query": False,
        "use_contextual_compressor": False,
        "use_conflict_detector": False,
        "use_cross_reference": False,
        "use_confidence_gate": False,
        "use_self_correction": False,
        "use_absence_prover": False,
        "use_semantic_cache": False,
        "use_output_contract": False,
        "use_cost_control": False,
    },
    PipelineVariant.ADVANCED: {
        "label": "Advanced RAG",
        "description": "Query rewriting + BGE reranker + adaptive chunking.",
        "use_hybrid_search": False,
        "use_reranker": True,
        "use_query_rewriting": True,
        "use_multi_query": False,
        "use_contextual_compressor": False,
        "use_conflict_detector": False,
        "use_cross_reference": False,
        "use_confidence_gate": False,
        "use_self_correction": False,
        "use_absence_prover": False,
        "use_semantic_cache": False,
        "use_output_contract": False,
        "use_cost_control": False,
    },
    PipelineVariant.HYBRID: {
        "label": "Hybrid RAG",
        "description": "BM25 + FAISS + RRF fusion with reranker.",
        "use_hybrid_search": True,
        "use_reranker": True,
        "use_query_rewriting": True,
        "use_multi_query": False,
        "use_contextual_compressor": False,
        "use_conflict_detector": False,
        "use_cross_reference": False,
        "use_confidence_gate": False,
        "use_self_correction": False,
        "use_absence_prover": False,
        "use_semantic_cache": False,
        "use_output_contract": False,
        "use_cost_control": False,
    },
    PipelineVariant.CORRECTIVE: {
        "label": "Corrective RAG",
        "description": "Hybrid + confidence gate + absence prover + self-correction loop.",
        "use_hybrid_search": True,
        "use_reranker": True,
        "use_query_rewriting": True,
        "use_multi_query": True,
        "use_contextual_compressor": True,
        "use_conflict_detector": True,
        "use_cross_reference": True,
        "use_confidence_gate": True,
        "use_self_correction": True,
        "use_absence_prover": True,
        "use_semantic_cache": False,
        "use_output_contract": False,
        "use_cost_control": False,
    },
    PipelineVariant.AGENTIC: {
        "label": "Agentic RAG",
        "description": "LangGraph retrieval agent with tool-calling, replanning, and multi-step reasoning.",
        "use_hybrid_search": True,
        "use_reranker": True,
        "use_query_rewriting": True,
        "use_multi_query": True,
        "use_contextual_compressor": True,
        "use_conflict_detector": True,
        "use_cross_reference": True,
        "use_confidence_gate": True,
        "use_self_correction": True,
        "use_absence_prover": True,
        "use_semantic_cache": False,
        "use_output_contract": True,
        "use_cost_control": False,
    },
    PipelineVariant.FULL_RAGOPS: {
        "label": "Full RAGOps",
        "description": "Complete production pipeline with all stages enabled (cache, cost control, claim verification).",
        "use_hybrid_search": True,
        "use_reranker": True,
        "use_query_rewriting": True,
        "use_multi_query": True,
        "use_contextual_compressor": True,
        "use_conflict_detector": True,
        "use_cross_reference": True,
        "use_confidence_gate": True,
        "use_self_correction": True,
        "use_absence_prover": True,
        "use_semantic_cache": True,
        "use_output_contract": True,
        "use_cost_control": True,
    },
}


def get_variant_config(variant: PipelineVariant) -> Dict[str, Any]:
    """Return the configuration dict for a given pipeline variant."""
    return VARIANT_CONFIGS[variant]


def list_variants() -> list:
    """Return human-readable list of all variant labels."""
    return [
        {"name": v.value, "label": cfg["label"], "description": cfg["description"]}
        for v, cfg in VARIANT_CONFIGS.items()
    ]
