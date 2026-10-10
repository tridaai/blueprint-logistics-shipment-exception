"""OpenTelemetry trace export: follow a run node by node.

Request IDs correlate the JSON logs; they do not show where a
provider-mode run *spent its time*. When a customer runs their own
tracing stack (Jaeger, Tempo, an APM), this module is the seam
that feeds it: one span per graph node, one per provider call, and
one per diagnosis tool call, exported over OTLP/HTTP — **off by
default**, enabled by setting
``OTEL_EXPORTER_OTLP_ENDPOINT`` (the standard OpenTelemetry
environment variable) and installing the ``otel`` extra.

Span discipline (the privacy contract of this surface):

- Names and attributes carry **ids and counts only** — the
  shipment id, the node name, the backend and model names, token
  counts, call counts. Never shipment content: no origins,
  destinations, customer names, document text, prompts, or drafts.
  A trace is an operations artefact other teams can see; the
  shipment record is the only place shipment content lives.
- The run span (``shipment.run``) parents the node spans
  (``node.<name>``); a provider span (``provider.<backend>``)
  opened inside a node parents to that node, so the tree reads
  the way the run executed.

Without the endpoint configured — or without the SDK installed —
every span here is the OpenTelemetry API's no-op: the call sites
never branch, and a run with tracing off pays nothing but a
context-manager hop. :func:`configure_tracing` runs at API
startup (and lazily on first span use, so the CLI and demo honour
the same variable); it is idempotent and never raises — a broken
tracing setup must not take the agent down with it.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from .config import env_str, load_dotenv

logger = logging.getLogger("shipment_agent.tracing")

_configured = False
_export_active = False

#: The only attribute keys a span may carry. Anything else a call
#: site passes is dropped on the floor — the allowlist is the
#: enforcement of the no-content contract above, not a convention.
_ALLOWED_ATTRIBUTES = frozenset(
    {
        "shipment_id",
        "tenant_id",
        "node",
        "backend",
        "model",
        "operation",
        "input_tokens",
        "output_tokens",
        "max_tokens",
        "tool_calls",
        "tool",
        "input_count",
        "duration_ms",
        "status",
    }
)

try:  # the API is a declared dependency; the SDK is the otel extra
    from opentelemetry import trace as _trace

    _API_AVAILABLE = True
except Exception:  # pragma: no cover - API missing: spans are no-ops
    _trace = None  # type: ignore[assignment]
    _API_AVAILABLE = False


def tracing_export_active() -> bool:
    """Whether spans are actually being exported (SDK wired to an
    OTLP endpoint). False in the default posture — spans exist as
    API no-ops and cost nothing."""
    return _export_active


def configure_tracing() -> bool:
    """Wire the OTLP exporter when configured. Returns the export
    state; idempotent, and never raises.

    Reads ``OTEL_EXPORTER_OTLP_ENDPOINT`` (and ``OTEL_SERVICE_NAME``,
    default ``shipment-agent``). With no endpoint set the process
    stays in the default posture: no provider is installed and the
    API's no-op tracer serves every span. With an endpoint but no
    SDK (the ``otel`` extra not installed) the misconfiguration is
    logged once, loudly, and the agent runs untraced rather than
    not at all.
    """
    global _configured, _export_active
    if _configured:
        return _export_active
    _configured = True
    load_dotenv()
    endpoint = env_str("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False
    if not _API_AVAILABLE:
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry-api "
            "is not installed; tracing export disabled"
        )
        return False
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create(
                {"service.name": env_str("OTEL_SERVICE_NAME") or "shipment-agent"}
            )
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
        _trace.set_tracer_provider(provider)
    except Exception as exc:  # missing extra, bad endpoint, double-set
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but the tracing "
            "exporter could not be wired (%s); install the 'otel' "
            "extra. Tracing export disabled.",
            exc,
        )
        return False
    _export_active = True
    logger.info("OpenTelemetry trace export on: %s", endpoint)
    return True


def _tracer():
    if not _configured:
        configure_tracing()
    if _trace is None:
        return None
    return _trace.get_tracer("shipment_agent")


def _clean(attributes: dict | None) -> dict:
    """The allowlist filter: ids and counts in, everything else out."""
    if not attributes:
        return {}
    return {
        key: value
        for key, value in attributes.items()
        if key in _ALLOWED_ATTRIBUTES and isinstance(value, (str, int, float, bool))
    }


class _SpanHandle:
    """The sliver of a span call sites may touch after opening:
    record the counts the response revealed. Safe on a no-op span
    and when the API is absent entirely (``span`` is then None)."""

    def __init__(self, span) -> None:
        self._span = span

    def set_attribute(self, key: str, value) -> None:
        if self._span is None:
            return
        cleaned = _clean({key: value})
        if cleaned:
            try:
                self._span.set_attribute(key, cleaned[key])
            except Exception:
                pass  # a span must never break the work it observes

    def set_counts(self, **counts) -> None:
        for key, value in counts.items():
            if value is not None:
                self.set_attribute(key, value)


@contextmanager
def span(name: str, attributes: dict | None = None) -> Iterator[_SpanHandle]:
    """One span, whatever the tracing posture.

    The OpenTelemetry context manager records exceptions and marks
    the span errored on the way out; with no provider wired it is
    the API's non-recording span and this costs a hop.
    """
    tracer = _tracer()
    if tracer is None:
        yield _SpanHandle(None)
        return
    with tracer.start_as_current_span(
        name, attributes=_clean(attributes)
    ) as otel_span:
        yield _SpanHandle(otel_span)


def run_span(shipment_id: str):
    """The run's root span: one analysis, shipment id only."""
    return span("shipment.run", {"shipment_id": shipment_id})


def node_span(name: str, shipment_id: str):
    """One graph node's span, parented to the active run span."""
    return span(
        f"node.{name}", {"node": name, "shipment_id": shipment_id}
    )


def provider_span(backend: str, model: str | None, operation: str):
    """One provider call's span (a completion, a tool-loop request,
    an embeddings request), parented to the active node span."""
    return span(
        f"provider.{backend}",
        {"backend": backend, "model": model or "", "operation": operation},
    )


def tool_span(tool: str):
    """One diagnosis tool call's span, parented to the active span
    (the diagnose node's, across its provider calls).

    The provider spans show the diagnosis *waiting on the model*;
    these show what the agent did between calls — which lookups it
    ran (``search_policies``, ``lane_history``, …) and how long
    each took. The dispatch site sets ``duration_ms`` (measured
    around the handler) and ``status`` when the call returns. The
    privacy contract holds: the tool's NAME and timing only —
    never its arguments or its results."""
    return span(f"tool.{tool}", {"tool": tool, "operation": "tool_call"})
