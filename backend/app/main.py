import logging
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv(override=False)  # real environment variables always win over .env

from app.utils import httpx_patch  # noqa: E402,F401  (must run before LLM SDKs import httpx)

from fastapi import Depends, FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db import engine, init_db  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ragops")

from app.auth import routes as auth_routes  # noqa: E402
from app.auth.deps import get_current_user  # noqa: E402
from app.mcp.http_server import build_app as build_mcp_app, mcp as mcp_server  # noqa: E402
from app.rag import (  # noqa: E402
    analytics_routes, api_key_routes, chat_routes, config_routes, database_routes, ingest_routes,
    inspector_routes, play_routes, project_routes,
)
from app.routes import api_analytics, benchmark_routes, query, workspace_routes  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    from app.auth.bootstrap import bootstrap_admin
    bootstrap_admin()
    from app.auth.bootstrap import bootstrap_demo
    bootstrap_demo()
    from app.mcp.registry_init import register_tools
    register_tools()
    if settings.run_worker:
        from app.jobs.queue import start_workers
        start_workers(settings.worker_threads)
    if settings.preload_models:
        from app.services.model_registry import preload
        threading.Thread(target=preload, name="model-preload", daemon=True).start()
    async with mcp_server.session_manager.run():
        yield
    if settings.run_worker:
        from app.jobs.queue import stop_workers
        stop_workers()


app = FastAPI(title="RAGOps API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    # Log the details server-side; never leak stack traces or internals to clients.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(auth_routes.router)
app.include_router(project_routes.router)
app.include_router(config_routes.router)
app.include_router(ingest_routes.router)
app.include_router(chat_routes.router)
app.include_router(query.router)
app.include_router(inspector_routes.router, prefix="/rag")
app.include_router(play_routes.router)
app.include_router(analytics_routes.router)
app.include_router(api_analytics.router)
app.include_router(benchmark_routes.router)
app.include_router(database_routes.router)
app.include_router(api_key_routes.router)
app.include_router(workspace_routes.router)
app.mount("/mcp", build_mcp_app())


@app.get("/models/catalog")
def models_catalog(_=Depends(get_current_user)):
    """Models available right now with the caller's workspace keys, per provider."""
    from app.services.llm_factory import model_catalog
    return {"providers": model_catalog()}


@app.get("/ping")
def ping():
    """Liveness without touching the database: use this for uptime monitors and the
    platform health check, so a scale-to-zero database (Neon) can still sleep."""
    return {"status": "ok"}


@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    status = "ok" if db_ok else "degraded"
    return JSONResponse(status_code=200 if db_ok else 503, content={"status": status, "database": db_ok})


@app.get("/")
def read_root():
    return {"message": "Welcome to RAGOps API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
