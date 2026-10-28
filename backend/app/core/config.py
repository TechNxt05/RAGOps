"""
Central runtime settings.

Every tunable that used to be scattered across modules as a bare os.getenv()
with an unsafe default lives here, so deployments (Render, Oracle VM, local)
are configured from one place via environment variables.
"""

import logging
import os
import secrets
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    def __init__(self) -> None:
        self.environment = os.getenv("ENVIRONMENT", "development").lower()

        # --- Database -------------------------------------------------------
        self.database_url = os.getenv("DATABASE_URL", "").strip()
        self.db_echo = _bool("DB_ECHO", False)

        # --- Auth -----------------------------------------------------------
        # SECRET_KEY is canonical; JWT_SECRET is accepted for older .env files.
        secret = os.getenv("SECRET_KEY") or os.getenv("JWT_SECRET")
        self.secret_key_configured = bool(secret)
        if not secret:
            # A random per-process key is safer than a well-known default:
            # tokens simply stop validating after a restart.
            secret = secrets.token_urlsafe(48)
            logger.warning(
                "SECRET_KEY is not set: using a random per-process key. "
                "Logins will reset on every restart and encrypted DB credentials "
                "cannot be stored. Set SECRET_KEY in the environment."
            )
        self.secret_key = secret
        self.access_token_expire_minutes = _int("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 12)
        # Self-registration always creates CLIENT users; the very first account
        # becomes ADMIN only when this is enabled (bootstrap of a fresh install).
        self.first_user_is_admin = _bool("FIRST_USER_IS_ADMIN", True)
        self.allow_self_registration = _bool("ALLOW_SELF_REGISTRATION", True)

        # --- CORS -----------------------------------------------------------
        default_origins = [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:3007",
        ]
        extra = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
        self.cors_origins: List[str] = default_origins + extra
        self.cors_origin_regex = os.getenv(
            "CORS_ORIGIN_REGEX", r"https://.*\.(vercel\.app|pages\.dev|workers\.dev)"
        )

        # --- Ingestion ------------------------------------------------------
        self.max_upload_mb = _int("MAX_UPLOAD_MB", 25)
        self.max_files_per_upload = _int("MAX_FILES_PER_UPLOAD", 20)
        self.run_worker = _bool("RUN_WORKER", True)
        self.worker_threads = max(1, _int("WORKER_THREADS", 2))
        # Jobs are enqueued by this same process, which wakes the workers instantly.
        # The idle poll is only a safety net; keeping it rare lets a serverless
        # database (Neon free: 100 compute-hours/month) scale to zero.
        self.worker_idle_poll_seconds = max(5, _int("WORKER_IDLE_POLL_SECONDS", 3600))
        # Simultaneous local model inferences (bounds peak memory on small hosts).
        self.model_concurrency = max(1, _int("MODEL_CONCURRENCY", 1))

        # --- Models ---------------------------------------------------------
        self.default_embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        # "onnx" runs the local models with ONNX Runtime (fastembed): no PyTorch, ~5x
        # less memory, fits a 512 MB free host. "torch" uses sentence-transformers.
        self.model_backend = os.getenv("MODEL_BACKEND", "onnx").strip().lower()
        default_reranker = "Xenova/ms-marco-MiniLM-L-6-v2" if self.model_backend == "onnx" else "BAAI/bge-reranker-base"
        self.reranker_model = os.getenv("RERANKER_MODEL", default_reranker)
        # Cap on candidates sent to the cross-encoder (CPU-bound on small hosts).
        self.rerank_max_candidates = _int("RERANK_MAX_CANDIDATES", 20 if self.model_backend == "onnx" else 30)
        self.model_threads = max(1, _int("MODEL_THREADS", 1))
        self.model_cache_dir = os.getenv("MODEL_CACHE_DIR") or None
        # ONNX Runtime's memory arena pre-allocates aggressively: off halves peak model
        # memory (measured 491 -> 226 MB) for ~15% slower inference.
        self.onnx_memory_arena = _bool("ONNX_MEMORY_ARENA", False)
        self.embed_batch_size = max(1, _int("EMBED_BATCH_SIZE", 16))
        self.preload_models = _bool("PRELOAD_MODELS", True)

        # --- Rate limiting (per user, per minute) ----------------------------
        self.rate_limit_chat_per_min = _int("RATE_LIMIT_CHAT_PER_MIN", 30)
        self.rate_limit_upload_per_min = _int("RATE_LIMIT_UPLOAD_PER_MIN", 20)

        # --- External database connector ------------------------------------
        self.db_connector_allow_private_hosts = _bool("DB_CONNECTOR_ALLOW_PRIVATE_HOSTS", False)
        self.db_connector_row_limit = _int("DB_CONNECTOR_ROW_LIMIT", 200)
        self.db_connector_timeout_ms = _int("DB_CONNECTOR_TIMEOUT_MS", 15000)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
