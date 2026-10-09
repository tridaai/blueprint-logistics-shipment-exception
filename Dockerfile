FROM python:3.12-slim

# uv for the locked install (single static binary copied from the official image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md constraints.txt uv.lock ./
COPY src ./src
COPY data ./data
# Default build: default dependencies only. The local-stack compose
# override passes UV_EXTRA_ARGS="--extra llm --extra vectordb" to add
# the provider SDKs and the Chroma store to the image.
ARG UV_EXTRA_ARGS=""
RUN uv sync --frozen $UV_EXTRA_ARGS

ENV PYTHONPATH=/app/src
ENV MODEL_BACKEND=mock
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "shipment_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
