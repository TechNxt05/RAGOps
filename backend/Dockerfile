FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_CACHE_DIR=/app/.models \
    MALLOC_ARENA_MAX=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-torch.txt ./
RUN pip install -r requirements.txt

# Default: ONNX models (fits 512 MB hosts). For the PyTorch backend with
# bge-reranker-base on a bigger machine:  --build-arg MODEL_BACKEND=torch
ARG MODEL_BACKEND=onnx
ENV MODEL_BACKEND=$MODEL_BACKEND
RUN if [ "$MODEL_BACKEND" = "torch" ]; then \
      pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu && \
      pip install -r requirements-torch.txt; \
    fi

# Optional layout-aware PDF parsing (tables, headings, OCR); needs PyTorch and RAM.
ARG INSTALL_DOCLING=false
RUN if [ "$INSTALL_DOCLING" = "true" ]; then pip install docling; fi

RUN useradd -m -u 1000 app && mkdir -p /app/.models && chown -R app:app /app
USER app

ARG BAKE_EMBEDDING_MODELS=all-MiniLM-L6-v2
COPY --chown=app:app . .
RUN BAKE_EMBEDDING_MODELS=$BAKE_EMBEDDING_MODELS python download_models.py
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8000}/ping || exit 1

# One process: models are loaded once and shared by the request thread pool and
# the ingestion workers. $PORT is set by Render; 8000 elsewhere.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips '*' --timeout-keep-alive 75"]
