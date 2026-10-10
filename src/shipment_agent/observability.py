"""Runtime observability: structured JSON logs with request IDs.

Production services do not get print statements and plain-text logs:
every log line the API emits is one JSON object on stdout (the
platform's log pipeline — CloudWatch, Loki, journald — parses it),
carrying a ``request_id`` that is also returned to the caller as the
``X-Request-ID`` header, so a support conversation can name the exact
request. The ID comes from the caller's header when present
(end-to-end tracing across the customer's systems) and is generated
otherwise.

Wired up in ``api.py``: a middleware assigns the ID, times the
request, and writes one access line per request; the lifespan
configures the formatter at startup. The CLI and demo keep their
human-readable output — this module serves the API surface.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_ctx: ContextVar[str | None] = ContextVar(
    "shipment_request_id", default=None
)


def new_request_id() -> str:
    return uuid.uuid4().hex


class JsonLogFormatter(logging.Formatter):
    """One JSON object per record; known extras ride along."""

    _EXTRA_KEYS = ("method", "path", "status", "duration_ms", "event")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        for key in self._EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_json_logging(level: str = "INFO") -> None:
    """Point the root logger at stdout with the JSON formatter.

    Idempotent: repeat calls (lifespan + imports) replace the marked
    handler instead of stacking duplicates.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_shipment_json", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    handler._shipment_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
