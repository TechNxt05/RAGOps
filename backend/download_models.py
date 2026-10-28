"""Download the local open-source models at build time so the first request
after a deploy (or a wake-up from sleep) doesn't wait on a download.

Uses the configured backend (MODEL_BACKEND, default "onnx") and MODEL_CACHE_DIR,
so the files land where the running app will look for them.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings  # noqa: E402

EMBEDDING_MODELS = [m.strip() for m in os.getenv("BAKE_EMBEDDING_MODELS", settings.default_embedding_model).split(",") if m.strip()]

if settings.model_backend == "onnx":
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    from app.services.model_registry import _fastembed_name

    for name in EMBEDDING_MODELS:
        print(f"Downloading embedding model {name} (ONNX)...")
        TextEmbedding(_fastembed_name(name), cache_dir=settings.model_cache_dir)
    if settings.reranker_model.lower() not in ("", "none", "off"):
        print(f"Downloading reranker {settings.reranker_model} (ONNX)...")
        TextCrossEncoder(settings.reranker_model, cache_dir=settings.model_cache_dir)
else:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    for name in EMBEDDING_MODELS:
        print(f"Downloading embedding model {name} (torch)...")
        SentenceTransformer(name)
    print(f"Downloading reranker {settings.reranker_model} (torch)...")
    CrossEncoder(settings.reranker_model, max_length=512)

print("Models cached.")
