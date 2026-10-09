FROM python:3.12-slim

# uv for the locked install (single static binary copied from the official image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md constraints.txt uv.lock ./
COPY src ./src
COPY data ./data
RUN uv sync --frozen

ENV PYTHONPATH=/app/src
ENV MODEL_BACKEND=mock
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "shipment_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
