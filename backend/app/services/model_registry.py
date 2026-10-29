"""
Process-wide model registry.

Embedding models and the cross-encoder reranker are expensive to load (seconds,
hundreds of MB). Previously a fresh embedding model was built for every request.
Here each model is loaded exactly once per process and shared, guarded by a lock
so concurrent first requests don't load it twice.
"""

import logging
import math
import threading
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

from app.core import llm_keys
from app.core.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_embedders: Dict[str, object] = {}
_dims: Dict[str, int] = {}
_reranker = None
_reranker_failed = False
# Bounds simultaneous local inferences: on a 0.1-CPU / 512 MB host parallel runs add
# memory without adding throughput.
_inference_slots = threading.BoundedSemaphore(settings.model_concurrency)

# Tests can install a deterministic embedder: fn(texts) -> vectors
_embedding_override: Optional[Callable[[List[str]], List[List[float]]]] = None


def set_embedding_override(fn: Optional[Callable[[List[str]], List[List[float]]]]) -> None:
    global _embedding_override
    _embedding_override = fn
    _dims.clear()


def normalize_model_name(name: Optional[str]) -> str:
    if not name or name in ("huggingface", "google-embedding-001"):
        return settings.default_embedding_model
    return name


def is_remote_model(name: str) -> bool:
    lowered = name.lower()
    return "gemini" in lowered or lowered.startswith("models/")


class _OverrideEmbedder:
    def embed_documents(self, texts):
        return _embedding_override(list(texts))

    def embed_query(self, text):
        return _embedding_override([text])[0]


def get_embedder(model_name: Optional[str] = None):
    """Return a LangChain-compatible embedder (embed_documents / embed_query)."""
    if _embedding_override is not None:
        return _OverrideEmbedder()
    name = normalize_model_name(model_name)
    cached = _embedders.get(name)
    if cached is not None:
        return cached
    with _lock:
        cached = _embedders.get(name)
        if cached is not None:
            return cached
        if is_remote_model(name):
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            model_id = name if name.startswith("models/") else f"models/{name}"
            embedder = GoogleGenerativeAIEmbeddings(model=model_id, google_api_key=llm_keys.get_key("google"))
        elif settings.model_backend == "onnx":
            logger.info("Loading embedding model %s (ONNX)", name)
            embedder = _OnnxEmbedder(name)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            logger.info("Loading embedding model %s (torch)", name)
            embedder = HuggingFaceEmbeddings(
                model_name=name,
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )
        _embedders[name] = embedder
        return embedder


_lean_ort_patched = False


def _lean_onnx_sessions() -> None:
    """Make every ONNX Runtime session fastembed creates skip the memory arena."""
    global _lean_ort_patched
    if _lean_ort_patched or settings.onnx_memory_arena:
        return
    import onnxruntime as ort
    base = ort.SessionOptions

    class _LeanSessionOptions(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.enable_cpu_mem_arena = False
            self.enable_mem_pattern = False

    ort.SessionOptions = _LeanSessionOptions
    _lean_ort_patched = True


def _fastembed_name(name: str) -> str:
    # sentence-transformers short names -> the hub ids fastembed lists
    return f"sentence-transformers/{name}" if "/" not in name else name


class _OnnxEmbedder:
    """Same model weights as the sentence-transformers path, run with ONNX Runtime
    (no PyTorch). Vectors are L2-normalised so both backends are interchangeable."""

    def __init__(self, name: str):
        _lean_onnx_sessions()
        from fastembed import TextEmbedding
        self._model = TextEmbedding(_fastembed_name(name), threads=settings.model_threads,
                                    cache_dir=settings.model_cache_dir)

    @staticmethod
    def _norm(vec) -> List[float]:
        n = math.sqrt(sum(float(x) * float(x) for x in vec)) or 1.0
        return [float(x) / n for x in vec]

    def embed_documents(self, texts):
        with _inference_slots:
            return [self._norm(v) for v in self._model.embed(list(texts), batch_size=settings.embed_batch_size)]

    def embed_query(self, text):
        with _inference_slots:
            return self._norm(next(iter(self._model.embed([text]))))


def embedding_dim(model_name: Optional[str] = None) -> int:
    name = normalize_model_name(model_name)
    if name not in _dims:
        _dims[name] = len(get_embedder(name).embed_query("dimension probe"))
    return _dims[name]


def embed_documents(texts: List[str], model_name: Optional[str] = None, batch_size: int = 64) -> List[List[float]]:
    embedder = get_embedder(model_name)
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embedder.embed_documents(texts[start:start + batch_size]))
    return [list(map(float, v)) for v in vectors]


_query_cache: "OrderedDict[tuple, List[float]]" = OrderedDict()
_QUERY_CACHE_SIZE = 512


def embed_query(text: str, model_name: Optional[str] = None) -> List[float]:
    """Query embeddings are memoised: one question is embedded for the cache lookup,
    vector search and relevance scoring, and identical strings give identical vectors."""
    key = (normalize_model_name(model_name), text)
    if _embedding_override is None:
        hit = _query_cache.get(key)
        if hit is not None:
            _query_cache.move_to_end(key)
            return hit
    vec = [float(x) for x in get_embedder(model_name).embed_query(text)]
    if _embedding_override is None:
        _query_cache[key] = vec
        if len(_query_cache) > _QUERY_CACHE_SIZE:
            _query_cache.popitem(last=False)
    return vec


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class _OnnxReranker:
    def __init__(self, name: str):
        _lean_onnx_sessions()
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self._model = TextCrossEncoder(name, threads=settings.model_threads, cache_dir=settings.model_cache_dir)

    def logits(self, query: str, texts: List[str]) -> List[float]:
        with _inference_slots:
            return [float(x) for x in self._model.rerank(query, texts, batch_size=8)]


class _TorchReranker:
    def __init__(self, name: str):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(name, max_length=512)

    def logits(self, query: str, texts: List[str]) -> List[float]:
        import torch
        out = self._model.predict([(query, t) for t in texts], activation_fn=torch.nn.Identity(), batch_size=16)
        return [float(x) for x in out]


def get_reranker():
    """Cross-encoder reranker for the configured backend (ONNX by default).
    Returns None if it cannot be loaded."""
    global _reranker, _reranker_failed
    if _reranker is not None or _reranker_failed:
        return _reranker
    with _lock:
        if _reranker is not None or _reranker_failed:
            return _reranker
        if settings.reranker_model.lower() in ("", "none", "off"):
            # Memory safety valve for tiny hosts: relevance then comes from cosine.
            logger.info("Reranker disabled by RERANKER_MODEL=%s", settings.reranker_model)
            _reranker_failed = True
            return None
        try:
            logger.info("Loading reranker %s (%s)", settings.reranker_model, settings.model_backend)
            cls = _OnnxReranker if settings.model_backend == "onnx" else _TorchReranker
            _reranker = cls(settings.reranker_model)
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.error("Reranker unavailable (%s); falling back to retrieval order.", exc)
            _reranker_failed = True
    return _reranker


# Per-model calibration of cross-encoder logits -> relevance in [0, 1]:
#   relevance = clamp((logit - floor) / span). Measured on labelled pairs:
# * bge-reranker-base: relevant -2.3..+5.9, irrelevant -10.2..-7.9
#   -> relevant >= 0.48, irrelevant <= 0.01
# * ms-marco-MiniLM-L-6-v2: irrelevant tightly at -11.45..-11.15, relevant mostly
#   -6..+7.4 (a few heavy paraphrases lower, which the cosine signal covers)
#   -> irrelevant = 0
# Raw sigmoids are too skewed to threshold, hence the linear logit map.
RELEVANCE_CALIBRATION = {
    "BAAI/bge-reranker-base": (-8.0, 12.0),
    "Xenova/ms-marco-MiniLM-L-6-v2": (-10.5, 14.0),
}


def logit_to_relevance(logit: float, model: Optional[str] = None) -> float:
    floor, span = RELEVANCE_CALIBRATION.get(model or settings.reranker_model, (-8.0, 12.0))
    return max(0.0, min(1.0, (float(logit) - floor) / span))


def cosine_to_relevance(cos: float) -> float:
    """Fallback when the reranker is off: MiniLM cosine ~0.35-0.7 for relevant
    passages, ~0-0.2 for unrelated ones."""
    return max(0.0, min(1.0, (float(cos) - 0.1) / 0.5))


def rerank_scores(query: str, texts: List[str]) -> Optional[List[float]]:
    """Calibrated relevance in [0, 1] from the cross-encoder, or None when no
    reranker is available. Ordering is identical to the raw model scores."""
    if not texts:
        return []
    model = get_reranker()
    if model is None:
        return None
    return [logit_to_relevance(x) for x in model.logits(query, texts)]


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def preload() -> None:
    """Warm the default models at startup so the first user request isn't slow."""
    try:
        embedding_dim(settings.default_embedding_model)
        get_reranker()
        from app.services import llm_factory
        chain = llm_factory.model_chain(None)
        llm_factory.build_with_fallbacks(chain, temperature=0.0, max_tokens=2048, fast=True)
    except Exception as exc:  # pragma: no cover
        logger.warning("Model preload failed: %s", exc)
