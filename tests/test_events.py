"""Run-event streaming: order, completeness, durations, and surfaces.

The graph emits structured events through an EventSink (events.py):
run_started → per-node node_started/node_finished (with durations) →
semantic events (tool_called, guardrail_verdict, repair_attempted) →
run_completed. These tests pin that contract in the offline fallback
mode, plus the two streaming surfaces: the CLI's --stream flag and
the API's SSE endpoint. The non-stream paths are covered by the rest
of the suite, unchanged.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import shipment_agent.api as api_module
from shipment_agent import cli
from shipment_agent.events import CollectingSink
from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import load_sample_shipments, sample_shipment_models
from shipment_agent.schemas import ShipmentInput

PIPELINE_NODES = [
    "extract", "ingest", "classify", "retrieve", "diagnose", "options",
    "draft", "verify", "review", "validate", "human_approval",
]

# The graph's node set: the eleven trace-step nodes plus the two
# evidence fan-out branches (discrepancies, history), which run
# concurrently with retrieve and so interleave in the event stream —
# order is only asserted where the topology actually orders events.
GRAPH_NODES = set(PIPELINE_NODES) | {"discrepancies", "history"}


def _run_sample(index=0, sink=None, **kwargs):
    shipment = sample_shipment_models()[index]
    return run_shipment(
        shipment,
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        event_sink=sink,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Event sequence in the offline fallback mode
# ---------------------------------------------------------------------------

def test_event_sequence_is_complete_and_ordered():
    sink = CollectingSink()
    result = _run_sample(0, sink=sink)
    types = [e.type for e in sink.events]
    assert types[0] == "run_started"
    assert types[-1] == "run_completed"
    assert sink.events[-1].detail["status"] == "awaiting_approval"
    assert result.approval_status == "awaiting_approval"

    started = [e.node for e in sink.events if e.type == "node_started"]
    finished = [e.node for e in sink.events if e.type == "node_finished"]
    assert set(started) == GRAPH_NODES
    assert set(finished) == GRAPH_NODES
    # The order the topology does guarantee: intake before classify,
    # the evidence branches before diagnose, the gate last.
    assert max(finished.index(n) for n in ("extract", "ingest")) < finished.index("classify")
    assert max(
        finished.index(n) for n in ("retrieve", "discrepancies", "history")
    ) < finished.index("diagnose")
    assert finished[-1] == "human_approval"
    for event in sink.events:
        if event.type == "node_finished":
            assert event.duration_ms is not None and event.duration_ms >= 0
        assert event.shipment_id == result.shipment_id

    verdicts = [e for e in sink.events if e.type == "guardrail_verdict"]
    assert len(verdicts) == 1
    assert verdicts[0].detail["passed"] is True


def test_guardrail_failure_and_repair_events_on_syn1013():
    samples = load_sample_shipments()
    index = next(
        i for i, s in enumerate(samples) if s["shipment_id"] == "SYN-1013"
    )
    sink = CollectingSink()
    _run_sample(index, sink=sink)
    verdict = next(e for e in sink.events if e.type == "guardrail_verdict")
    assert verdict.detail["passed"] is False
    assert verdict.detail["errors"]
    repairs = [e for e in sink.events if e.type == "repair_attempted"]
    assert len(repairs) == 1
    assert repairs[0].detail["attempts"] == 1
    assert repairs[0].detail["repaired"] is False


def test_tool_called_events_from_the_diagnosis_tool_loop():
    class ToolCallingBackend(MockModelBackend):
        name = "fake-tools"

        def diagnose(self, context):  # the tools path preempts this
            return {"root_cause": "unused", "summary": "unused"}

        def diagnose_with_tools(self, context, specs, dispatch, max_calls):
            summary = dispatch("shipment_facts", {})
            return {
                "root_cause": "Weather hold at the hub.",
                "summary": "Delay from a weather hold.",
                "tool_calls": [{"name": "shipment_facts", "summary": str(summary)}],
            }

    shipment = ShipmentInput.model_validate(load_sample_shipments()[0])
    sink = CollectingSink()
    run_shipment(
        shipment,
        backend=ToolCallingBackend(),
        retriever=KeywordRetriever(),
        event_sink=sink,
    )
    tool_events = [e for e in sink.events if e.type == "tool_called"]
    assert len(tool_events) == 1
    assert tool_events[0].node == "diagnose"
    assert tool_events[0].detail["tool"] == "shipment_facts"


def test_trace_steps_carry_node_durations():
    result = _run_sample(0, sink=CollectingSink())
    assert [s.name for s in result.trace] == PIPELINE_NODES
    for step in result.trace:
        assert step.duration_ms is not None and step.duration_ms >= 0


def test_run_failed_event_on_a_hard_provider_error():
    class ExplodingBackend(MockModelBackend):
        name = "exploding"

        def draft_customer_update(self, context):
            raise RuntimeError("provider exploded")

    sink = CollectingSink()
    with pytest.raises(RuntimeError, match="provider exploded"):
        run_shipment(
            sample_shipment_models()[0],
            backend=ExplodingBackend(),
            retriever=KeywordRetriever(),
            event_sink=sink,
        )
    assert sink.events[-1].type == "run_failed"
    assert "provider exploded" in sink.events[-1].detail["error"]
    # The draft node started but never finished; no run_completed.
    assert "run_completed" not in [e.type for e in sink.events]


# ---------------------------------------------------------------------------
# CLI --stream
# ---------------------------------------------------------------------------

def test_cli_stream_prints_events_then_the_result(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["shipment-agent", "--index", "0", "--stream"])
    cli.main()
    out = capsys.readouterr().out
    assert "[event] run_started" in out
    assert "[event] node_finished" in out
    assert "[event] guardrail_verdict" in out
    assert "[event] run_completed" in out
    # The usual rendered result still follows, ending at the gate.
    assert "awaiting_approval" in out


# ---------------------------------------------------------------------------
# API SSE endpoint
# ---------------------------------------------------------------------------

def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_sse_endpoint_streams_events_and_the_final_result():
    client = TestClient(api_module.app)
    shipment = load_sample_shipments()[0]
    with client.stream("POST", "/shipments/analyze/stream", json=shipment) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in response.iter_text())
    events = _parse_sse(body)
    assert events[0]["type"] == "run_started"
    node_finishes = [e["node"] for e in events if e["type"] == "node_finished"]
    assert set(node_finishes) == GRAPH_NODES
    assert node_finishes[-1] == "human_approval"
    final = events[-1]
    assert final["type"] == "run_completed"
    assert final["detail"]["status"] == "awaiting_approval"
    result = final["result"]
    assert result["shipment_id"] == shipment["shipment_id"]
    assert result["approval_status"] == "awaiting_approval"
    assert result["classification"]["exception_type"] == "delay"
    # The streamed result is the same shape the plain endpoint returns,
    # trace durations included.
    assert all(step["duration_ms"] is not None for step in result["trace"])
