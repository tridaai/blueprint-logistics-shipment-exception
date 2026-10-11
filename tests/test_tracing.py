"""OpenTelemetry trace export: span per node, span per provider call.

The mock exporter here is the SDK's in-memory exporter wired to a
real TracerProvider — the same provider shape production installs
behind OTEL_EXPORTER_OTLP_ENDPOINT, with the network exporter
swapped for a list. The assertions are about the *tree*: one run
span, the node spans parented to it, provider spans parented to
the node that made the call — and about the privacy contract:
attributes carry ids and counts only, never shipment content.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from shipment_agent import tracing
from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import MockModelBackend, OpenAIBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.schemas import ShipmentInput
from shipment_agent.tools_agent import DiagnosisToolBox

DELAY_SHIPMENT = {
    "shipment_id": "TRACE-1",
    "origin": "Zxqvania City",
    "destination": "Charlotte, NC",
    "customer_name": "Qwerty Konsignments LLC",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

SEQUENTIAL_NODES = [
    "extract",
    "ingest",
    "classify",
    "retrieve",
    "discrepancies",
    "history",
    "diagnose",
    "options",
    "draft",
    "verify",
    "review",
    "validate",
    "human_approval",
]


@pytest.fixture(scope="module")
def exporter():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    in_memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory))
    try:
        trace.set_tracer_provider(provider)
    except Exception:
        # A provider was already set earlier in the session; the
        # spans then flow to it and this module's exporter stays
        # empty — fail loudly rather than assert on nothing.
        raise RuntimeError("tracer provider already set by another test module")
    return in_memory


def _spans_for_trace(exporter, trace_id):
    return [s for s in exporter.get_finished_spans() if s.context.trace_id == trace_id]


def test_run_span_parents_every_node_span(exporter):
    exporter.clear()
    result = run_shipment(
        DELAY_SHIPMENT, backend=MockModelBackend(), evidence_mode="sequential"
    )
    assert result.shipment_id == "TRACE-1"
    spans = exporter.get_finished_spans()
    runs = [s for s in spans if s.name == "shipment.run"]
    assert len(runs) == 1
    run = runs[0]
    assert run.parent is None
    assert run.attributes["shipment_id"] == "TRACE-1"

    nodes = [s for s in spans if s.name.startswith("node.")]
    assert [s.name for s in nodes] == [f"node.{name}" for name in SEQUENTIAL_NODES]
    for node in nodes:
        assert node.context.trace_id == run.context.trace_id
        assert node.parent.span_id == run.context.span_id
        assert node.attributes["shipment_id"] == "TRACE-1"
        assert node.attributes["node"] in SEQUENTIAL_NODES


class _FakeCompletions:
    """A fake OpenAI transport, in the test_llm_backends idiom:
    the classify prompt gets a JSON suggestion, every other
    completion gets a clean draft, and usage rides along so the
    provider spans carry token counts."""

    def create(self, model=None, max_tokens=None, messages=None):
        system = " ".join(messages[0]["content"].split())
        if "classifying a shipment exception" in system:
            text = (
                '{"exception_type": "delay", "severity": "high",'
                ' "confidence": 0.9, "rationale": "Weather hold."}'
            )
        else:
            text = (
                "Subject: Update on shipment TRACE-1: delay\n\n"
                "Dear Qwerty Konsignments LLC,\n\n"
                "Your shipment is delayed by a weather hold. "
                "The next update will arrive within one business day."
            )
        usage = SimpleNamespace(prompt_tokens=13, completion_tokens=5)
        message = SimpleNamespace(content=text)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=usage
        )


def test_full_run_tree_nests_provider_spans_under_their_nodes(
    exporter, monkeypatch
):
    """A provider-mode run, end to end: the same run span and the
    same ordered node spans as the offline tree, plus a provider
    span for the provider calls, each parented to the node that
    made it — the tree a customer's tracing stack would show."""
    from shipment_agent.model_backends import OpenAIBackend

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions())
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **kwargs: fake_client),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "shipment_agent.model_backends.load_dotenv", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "shipment_agent.retriever.load_dotenv", lambda *a, **k: None
    )
    backend = OpenAIBackend()

    exporter.clear()
    result = run_shipment(
        DELAY_SHIPMENT, backend=backend, evidence_mode="sequential"
    )
    assert result.shipment_id == "TRACE-1"
    spans = exporter.get_finished_spans()
    runs = [s for s in spans if s.name == "shipment.run"]
    assert len(runs) == 1
    run = runs[0]
    nodes = [s for s in spans if s.name.startswith("node.")]
    assert [s.name for s in nodes] == [f"node.{name}" for name in SEQUENTIAL_NODES]
    node_ids = {s.context.span_id: s.name for s in nodes}

    providers = [s for s in spans if s.name == "provider.openai"]
    assert len(providers) >= 2
    for call in providers:
        assert call.context.trace_id == run.context.trace_id
        # Nested: the parent is one of this run's node spans —
        # never the run span itself, never a disconnected root.
        assert call.parent is not None
        assert call.parent.span_id in node_ids
        assert call.attributes["backend"] == "openai"
        assert call.attributes["operation"] == "chat_completion"
    parents = {node_ids[call.parent.span_id] for call in providers}
    # The classification cross-check and the draft are the two
    # calls every provider-mode run makes.
    assert "node.classify" in parents
    assert "node.draft" in parents
    # The plain completion calls stamp the usage the provider
    # reported onto their spans. (The diagnosis tool loop's
    # transport spans carry ids only — the loop accounts its
    # usage as a whole, so no per-call counts are claimed there.)
    for node_name in ("node.classify", "node.draft"):
        stamped = [
            c for c in providers if node_ids[c.parent.span_id] == node_name
        ]
        assert stamped, node_name
        for call in stamped:
            assert call.attributes["input_tokens"] == 13
            assert call.attributes["output_tokens"] == 5


def test_span_attributes_carry_ids_and_counts_never_content(exporter):
    exporter.clear()
    run_shipment(
        DELAY_SHIPMENT, backend=MockModelBackend(), evidence_mode="sequential"
    )
    spans = exporter.get_finished_spans()
    assert spans
    for s in spans:
        for key, value in (s.attributes or {}).items():
            assert key in tracing._ALLOWED_ATTRIBUTES, (s.name, key)
            assert "Zxqvania" not in str(value)
            assert "Qwerty" not in str(value)


def test_provider_span_parents_to_the_calling_node(exporter):
    exporter.clear()
    backend = OpenAIBackend.__new__(OpenAIBackend)
    backend.name = "openai"
    backend._model = "test-model"
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    message = SimpleNamespace(content="draft text")
    completions = SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=usage
        )
    )
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with tracing.node_span("draft", "TRACE-1"):
        text = backend._complete("system", "user")
    assert text == "draft text"

    spans = exporter.get_finished_spans()
    node = next(s for s in spans if s.name == "node.draft")
    provider = next(s for s in spans if s.name == "provider.openai")
    assert provider.parent.span_id == node.context.span_id
    assert provider.context.trace_id == node.context.trace_id
    assert provider.attributes["backend"] == "openai"
    assert provider.attributes["model"] == "test-model"
    assert provider.attributes["operation"] == "chat_completion"
    assert provider.attributes["input_tokens"] == 11
    assert provider.attributes["output_tokens"] == 7


class _ToolCallingBackend(MockModelBackend):
    """Mock drafting plus a scripted agentic diagnosis: two
    lookups through the toolbox, then the composed answer."""

    name = "stub-tools"

    def diagnose(self, context):
        return None

    def diagnose_with_tools(self, context, tool_specs, dispatch, max_tool_calls):
        dispatch("search_policies", {"query": "delay customer update"})
        dispatch("shipment_facts", {})
        return {
            "root_cause": "Weather hold at the hub.",
            "summary": "Carrier weather hold.",
            "tool_calls": [
                {"name": "search_policies", "summary": "2 policies"},
                {"name": "shipment_facts", "summary": "delay, 36.0h"},
            ],
        }


def test_tool_spans_parent_to_the_diagnose_node(exporter):
    exporter.clear()
    result = run_shipment(
        DELAY_SHIPMENT, backend=_ToolCallingBackend(), evidence_mode="sequential"
    )
    assert result.diagnosis.source == "llm"
    spans = exporter.get_finished_spans()
    diagnose = next(s for s in spans if s.name == "node.diagnose")
    tools = [s for s in spans if s.name.startswith("tool.")]
    assert [s.name for s in tools] == ["tool.search_policies", "tool.shipment_facts"]
    for tool in tools:
        # The lookups hang off the diagnose node — between its
        # provider calls in execution order — and carry the tool
        # name, its measured duration, and its outcome. Nothing
        # else: no arguments, no results (the privacy test above
        # sweeps every attribute against the allowlist).
        assert tool.parent.span_id == diagnose.context.span_id
        assert tool.context.trace_id == diagnose.context.trace_id
        assert tool.attributes["tool"] == tool.name.removeprefix("tool.")
        assert tool.attributes["operation"] == "tool_call"
        assert tool.attributes["status"] == "ok"
        assert isinstance(tool.attributes["duration_ms"], float)
        assert tool.attributes["duration_ms"] >= 0


def test_timed_provider_calls_keep_their_span_parent(exporter):
    """The resilience wrapper runs provider calls on a worker
    thread to hold the timeout; the thread inherits the caller's
    context, so spans opened inside the call still parent to the
    node span that was current when the call was made. Without
    the propagation every span inside a timed call — provider
    spans included — landed as a disconnected root and the tree
    lost exactly the calls it exists to explain."""
    from shipment_agent.resilience import ResilientBackend

    exporter.clear()
    toolbox = DiagnosisToolBox(
        shipment=ShipmentInput.model_validate(DELAY_SHIPMENT),
        classification={
            "exception_type": "delay",
            "severity": "high",
            "confidence": 0.9,
        },
        delay_hours=36.0,
        mismatches=[],
        retriever=KeywordRetriever(),
        priors=[],
    )
    backend = ResilientBackend(_ToolCallingBackend())
    with tracing.node_span("diagnose", "TRACE-1"):
        backend.diagnose_with_tools({}, [], toolbox.dispatch, 4)
    spans = exporter.get_finished_spans()
    node = next(s for s in spans if s.name == "node.diagnose")
    tools = [s for s in spans if s.name.startswith("tool.")]
    assert [s.name for s in tools] == ["tool.search_policies", "tool.shipment_facts"]
    for tool in tools:
        assert tool.parent.span_id == node.context.span_id


def test_tool_span_records_a_failed_call(exporter):
    exporter.clear()
    toolbox = DiagnosisToolBox(
        shipment=ShipmentInput.model_validate(DELAY_SHIPMENT),
        classification={
            "exception_type": "delay",
            "severity": "high",
            "confidence": 0.9,
        },
        delay_hours=36.0,
        mismatches=[],
        retriever=KeywordRetriever(),
        priors=[],
    )
    with tracing.node_span("diagnose", "TRACE-1"):
        with pytest.raises(ValueError):
            toolbox.dispatch("search_policies", {"query": "  "})
    spans = exporter.get_finished_spans()
    node = next(s for s in spans if s.name == "node.diagnose")
    tool = next(s for s in spans if s.name == "tool.search_policies")
    assert tool.parent.span_id == node.context.span_id
    assert tool.attributes["status"] == "error"
    assert tool.attributes["duration_ms"] >= 0


def test_tracing_is_off_by_default(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setattr(tracing, "_configured", False)
    monkeypatch.setattr(tracing, "_export_active", False)
    assert tracing.configure_tracing() is False
    assert tracing.tracing_export_active() is False


def test_configure_tracing_wires_the_otlp_exporter_when_configured():
    """In a subprocess (a tracer provider can only be set once per
    process): with the endpoint set and the exporter module present
    (a recording stand-in), configuration activates export and a
    span reaches the exporter."""
    program = r"""
import sys, types

recorded = []

class _StubExporter:
    def __init__(self, endpoint=None, **kwargs):
        recorded.append(("endpoint", endpoint))
    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult
        recorded.append(("export", [s.name for s in spans]))
        return SpanExportResult.SUCCESS
    def shutdown(self):
        pass

module = types.ModuleType("opentelemetry.exporter.otlp.proto.http.trace_exporter")
module.OTLPSpanExporter = _StubExporter
sys.modules["opentelemetry.exporter.otlp.proto.http.trace_exporter"] = module

import os
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://127.0.0.1:4318"

from shipment_agent import tracing
assert tracing.configure_tracing() is True, "configure should activate export"
assert tracing.tracing_export_active() is True
with tracing.run_span("TRACE-SUB"):
    pass
from opentelemetry import trace
trace.get_tracer_provider().force_flush()
flat = [item for entry in recorded for item in (entry[1] if isinstance(entry[1], list) else [])]
assert "shipment.run" in flat, recorded
assert ("endpoint", "http://127.0.0.1:4318") in recorded, recorded
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src"},
        cwd=str(__file__).rsplit("/tests/", 1)[0],
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_spans_are_harmless_noops_without_any_provider():
    """A fresh process with nothing configured: emitting the full
    span set must neither raise nor need the SDK's provider."""
    program = r"""
import os
os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
from shipment_agent import tracing
assert tracing.configure_tracing() is False
with tracing.run_span("TRACE-NOOP"):
    with tracing.node_span("draft", "TRACE-NOOP") as node:
        node.set_counts(input_tokens=3)
    with tracing.provider_span("openai", "m", "chat_completion") as call:
        call.set_counts(input_tokens=1, output_tokens=2)
    with tracing.tool_span("search_policies") as tool:
        tool.set_attribute("status", "ok")
        tool.set_attribute("duration_ms", 1.5)
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src"},
        cwd=str(__file__).rsplit("/tests/", 1)[0],
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
