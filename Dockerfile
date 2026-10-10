# Trida AI Blueprint — Logistics Shipment Exception Agent
#
# Multi-stage: the builder resolves the locked environment with uv;
# the runtime stage carries only the venv, the package, the packaged
# sample data, and the migrations — no build tooling, no test suite.
# The process runs as a non-root user and keeps NO local state: with
# DATABASE_URL / S3_* set (the compose stack sets them) records live
# in PostgreSQL, graph state in the Postgres checkpointer, vectors
# in pgvector, documents in object storage. Without them it serves
# on in-memory test doubles, which the /readiness payload names.
#
# Build:  docker build -t shipment-exception-agent .
# Run:    docker run --rm -p 8000:8000 \
#           -e MODEL_BACKEND=mock \
#           -e DATABASE_URL=postgresql://user:pass@host:5432/db \
#           shipment-exception-agent
#
# MODEL_BACKEND=mock runs fully offline. For a real model pass e.g.
#   -e MODEL_BACKEND=openai -e OPENAI_API_KEY=... -e OPENAI_MODEL=gpt-...
# or point OPENAI_BASE_URL at a self-hosted NVIDIA NIM (see README).
# UV_EXTRA_ARGS controls optional extras at build time:
#   docker build --build-arg UV_EXTRA_ARGS="--extra llm" .   (provider SDKs)
# The vectordb extra (Chroma client) is NOT needed for the
# pgvector production path.

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml README.md uv.lock constraints.txt ./
COPY src ./src
ARG UV_EXTRA_ARGS=""
# --frozen installs exactly uv.lock; --no-dev keeps the test
# toolchain (pytest & friends) out of the production image.
RUN uv sync --frozen --no-dev ${UV_EXTRA_ARGS}

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY data ./data
COPY migrations ./migrations

ENV PYTHONPATH=/app/src \
    PATH=/app/.venv/bin:$PATH \
    MODEL_BACKEND=mock \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

# Liveness only: /health answers without touching dependencies.
# (Readiness — including the database ping — is /readiness, for the
# orchestrator's readiness probe.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# uvicorn handles SIGTERM gracefully: stop accepting, drain
# in-flight requests, run the lifespan shutdown (pool close).
CMD ["uvicorn", "shipment_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
