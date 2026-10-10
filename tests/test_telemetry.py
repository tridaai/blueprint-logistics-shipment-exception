"""Run telemetry tests.

Every run reports what it used: model calls, provider-reported token
totals, wall-clock latency, and an estimated cost from the in-code
price table — on the result, and inside the claim packet. Mock mode
reports tokens/cost as None (no model ran; no fabricated numbers)
while keeping its call count and latency real.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import MockModelBackend, OpenAIBackend, estimate_cost_usd
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.schemas import ShipmentInput

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "RETRIEVER", "DIAGNOSIS_MAX_TOOL_CALLS",
]

DELAY_SHIPMENT = {
    "shipment_id": "TEL-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

CLASSIFY_TEXT = (
    '{"exception_type": "delay", "severity": "high", "confidence": 0.88,'
    ' "rationale": "Computed delay dominates."}'
)
DIAGNOSIS_TEXT = '{"root_cause": "Weather hold.", "summary": "Carrier weather hold."}'
OPTIONS_TEXT = '[{"kind": "expedite", "title": "Upgrade", "description": "Fly it."}]'
VERIFY_TEXT = '{"grounded": true, "issues": [], "summary": "Grounded."}'
DRAFT_TEXT = (
    "Subject: Update on shipment TEL-1: delay\n\n"
    "Dear Synthetic Customer,\n\n"
    "Your shipment TEL-1 is delayed. [POL-DELAY-01] applies. "
    "The next update will arrive within one business day."
)


class _UsageCompletions:
    """Every response reports the same usage: 100 in / 50 out."""

    def create(self, model=None, max_tokens=None, messages=None, tools=None):
        system = " ".join(messages[0]["content"].split())
        if "classifying a shipment exception" in system:
            text = CLASSIFY_TEXT
        elif "diagnosing the root cause" in system:
            text = DIAGNOSIS_TEXT
        elif "proposing recovery options" in system:
            text = OPTIONS_TEXT
        elif "verifying whether a drafted" in system:
            text = VERIFY_TEXT
        else:
            text = DRAFT_TEXT
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )


class UsageOpenAI:
    instances: list["UsageOpenAI"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=_UsageCompletions())
        UsageOpenAI.instances.append(self)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.model_backends.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.retriever.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.tools_agent.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def fake_openai(monkeypatch):
    UsageOpenAI.instances = []
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=UsageOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return UsageOpenAI


def _run(backend):
    return run_shipment(
        ShipmentInput.model_validate(DELAY_SHIPMENT),
        backend=backend,
        retriever=KeywordRetriever(),
    )


def test_provider_telemetry_aggregates_the_whole_run(fake_openai):
    result = _run(OpenAIBackend())
    t = result.telemetry
    assert t is not None
    assert t.backend == "openai"
    assert t.model == "gpt-4o-mini"
    # classify + diagnose + options + draft + verify = 5 model calls.
    assert t.model_calls == 5
    assert t.input_tokens == 500
    assert t.output_tokens == 250
    # gpt-4o-mini at (0.15, 0.60) per 1M tokens.
    assert t.estimated_cost_usd == pytest.approx(0.000225)
    assert t.latency_seconds > 0


def test_telemetry_rides_inside_the_claim_packet(fake_openai):
    result = _run(OpenAIBackend())
    packet_telemetry = result.draft.claim_packet["telemetry"]
    assert packet_telemetry["model_calls"] == result.telemetry.model_calls
    assert packet_telemetry["input_tokens"] == 500


def test_telemetry_is_a_delta_not_a_cumulative_total(fake_openai):
    backend = OpenAIBackend()
    first = _run(backend)
    second = _run(backend)  # same backend instance, second run
    assert second.telemetry.model_calls == first.telemetry.model_calls == 5
    assert second.telemetry.input_tokens == 500


def test_unknown_model_reports_cost_as_none_not_a_guess(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "mystery-model-9000")
    result = _run(OpenAIBackend())
    assert result.telemetry.input_tokens == 500
    assert result.telemetry.estimated_cost_usd is None


def test_mock_telemetry_is_honest_about_no_model():
    result = _run(MockModelBackend())
    t = result.telemetry
    assert t is not None
    assert t.backend == "mock"
    assert t.model is None
    assert t.input_tokens is None  # no fake numbers
    assert t.output_tokens is None
    assert t.estimated_cost_usd is None
    assert t.model_calls == 1  # the one template render — real
    assert t.latency_seconds >= 0


def test_estimate_cost_usd_table_behaviour():
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert estimate_cost_usd("llama3.1", 500, 500) == 0.0  # local = free, and known
    assert estimate_cost_usd("nope", 1, 1) is None
    assert estimate_cost_usd("gpt-4o-mini", None, 1) is None
    assert estimate_cost_usd(None, 1, 1) is None
