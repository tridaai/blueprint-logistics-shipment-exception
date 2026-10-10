"""Per-run token budget tests (RUN_TOKEN_BUDGET — a cost guardrail).

The pipeline tracks cumulative provider tokens via the backend's own
usage counters; once the budget is exceeded, the remaining provider
steps degrade to their deterministic/template paths with trace notes,
telemetry reports the budget accounting, and the run never hard-fails.
Unset, nothing changes. Mock mode is unaffected (it spends no tokens).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import MockModelBackend, OpenAIBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.schemas import ShipmentInput

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "RETRIEVER", "RUN_TOKEN_BUDGET",
    "DIAGNOSIS_MAX_TOOL_CALLS", "REVIEWER",
]

DELAY_SHIPMENT = {
    "shipment_id": "BUD-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

TWO_DOC_SHIPMENT = {
    **DELAY_SHIPMENT,
    "shipment_id": "BUD-2",
    "documents": [
        {
            "doc_type": "bol",
            "document_id": "BOL-BUD-2",
            "raw_text": "Bill of lading. Quantity: 120 units. Weight: 840 kg.",
            "fields": {"quantity_units": "120", "weight_kg": "840"},
        },
        {
            "doc_type": "invoice",
            "document_id": "INV-BUD-2",
            "raw_text": "Invoice. Quantity: 120 units. Weight: 840 kg.",
            "fields": {"quantity_units": "120", "weight_kg": "840"},
        },
    ],
}

CLASSIFY_TEXT = (
    '{"exception_type": "delay", "severity": "high", "confidence": 0.88,'
    ' "rationale": "Computed delay dominates."}'
)
DIAGNOSIS_TEXT = '{"root_cause": "Weather hold.", "summary": "Carrier weather hold."}'
OPTIONS_TEXT = '[{"kind": "expedite", "title": "Upgrade", "description": "Fly it."}]'
VERIFY_TEXT = '{"grounded": true, "issues": [], "summary": "Grounded."}'
REVIEW_TEXT = '{"verdict": "pass", "findings": []}'
DRAFT_TEXT = (
    "Subject: Update on shipment BUD-1: delay (provider draft)\n\n"
    "Dear Synthetic Customer,\n\n"
    "Your shipment BUD-1 is delayed. [POL-DELAY-01] applies. "
    "The next update will arrive within one business day."
)


class _UsageCompletions:
    """Every response reports the same usage: 100 in / 50 out = 150 tokens."""

    def create(self, model=None, max_tokens=None, messages=None, tools=None):
        system = " ".join(messages[0]["content"].split())
        if "extracting structured fields" in system:
            text = "{}"
        elif "classifying a shipment exception" in system:
            text = CLASSIFY_TEXT
        elif "diagnosing the root cause" in system:
            text = DIAGNOSIS_TEXT
        elif "proposing recovery options" in system:
            text = OPTIONS_TEXT
        elif "verifying whether a drafted" in system:
            text = VERIFY_TEXT
        elif "independent operations reviewer" in system:
            text = REVIEW_TEXT
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
    monkeypatch.setattr("shipment_agent.graph.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.reviewer.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.config.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def fake_openai(monkeypatch):
    UsageOpenAI.instances = []
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=UsageOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return UsageOpenAI


def _run(shipment, backend):
    return run_shipment(
        ShipmentInput.model_validate(shipment),
        backend=backend,
        retriever=KeywordRetriever(),
    )


def test_tiny_budget_degrades_every_step_after_the_first(fake_openai, monkeypatch):
    monkeypatch.setenv("RUN_TOKEN_BUDGET", "120")
    result = _run(DELAY_SHIPMENT, OpenAIBackend())
    # The first provider step (classify, 150 tokens) still ran…
    assert result.cross_check is not None
    assert result.cross_check.resolution == "agree"
    # …and everything after it degraded to its deterministic path.
    assert result.diagnosis is not None and result.diagnosis.source == "template"
    assert result.verification is not None and result.verification.source == "checklist"
    assert result.review is not None and result.review.source == "checklist"
    # Drafting fell back to the template renderer (its subject shape).
    assert result.draft.subject == "Update on shipment BUD-1: delay"
    assert "provider draft" not in result.draft.subject
    # The run completed normally — budget never hard-fails.
    assert result.validation.passed
    assert result.approval_status == "awaiting_approval"
    # Telemetry carries the accounting: one real provider call happened.
    t = result.telemetry
    assert t is not None and t.budget is not None
    assert t.budget.limit == 120
    assert t.budget.used == 150
    assert t.budget.exceeded is True
    assert t.model_calls == 1
    # The trace says which steps degraded, and the gate shows the budget.
    diagnose = next(s for s in result.trace if s.name == "diagnose")
    assert any("budget exceeded — template path" in d for d in diagnose.details)
    draft_step = next(s for s in result.trace if s.name == "draft")
    assert any("budget exceeded — template path" in d for d in draft_step.details)
    gate = next(s for s in result.trace if s.name == "human_approval")
    assert any("token budget: 150 of 120" in d for d in gate.details)


def test_extraction_degrades_mid_list_when_the_budget_runs_out(fake_openai, monkeypatch):
    monkeypatch.setenv("RUN_TOKEN_BUDGET", "120")
    result = _run(TWO_DOC_SHIPMENT, OpenAIBackend())
    # First document extracted by the LLM (150 tokens); the budget is
    # then spent, so the second document falls back to provided fields.
    assert [e.source for e in result.extractions] == ["llm", "provided"]
    # And classify (the next provider step) degraded too.
    assert result.cross_check is not None
    assert result.cross_check.resolution == "rules_only"
    assert result.telemetry is not None and result.telemetry.budget is not None
    assert result.telemetry.budget.exceeded is True


def test_generous_budget_changes_nothing_but_reports(fake_openai, monkeypatch):
    monkeypatch.setenv("RUN_TOKEN_BUDGET", "100000")
    result = _run(DELAY_SHIPMENT, OpenAIBackend())
    assert result.diagnosis is not None and result.diagnosis.source == "llm"
    assert result.draft.subject.endswith("(provider draft)")
    t = result.telemetry
    assert t is not None and t.budget is not None
    assert t.budget.limit == 100000
    assert t.budget.used == 900  # 6 provider calls x 150 tokens
    assert t.budget.exceeded is False
    for step in result.trace:
        assert not any("budget exceeded" in d for d in step.details)


def test_budget_unset_means_no_budget_anywhere(fake_openai):
    result = _run(DELAY_SHIPMENT, OpenAIBackend())
    assert result.telemetry is not None
    assert result.telemetry.budget is None
    assert result.diagnosis is not None and result.diagnosis.source == "llm"


def test_mock_mode_is_unaffected_by_a_budget(monkeypatch):
    monkeypatch.setenv("RUN_TOKEN_BUDGET", "10")
    result = _run(DELAY_SHIPMENT, MockModelBackend())
    assert result.classification.exception_type.value == "delay"
    assert result.diagnosis is not None and result.diagnosis.source == "template"
    t = result.telemetry
    assert t is not None and t.budget is not None
    assert t.budget.used == 0  # the mock spends no tokens, ever
    assert t.budget.exceeded is False
    for step in result.trace:
        assert not any("budget exceeded" in d for d in step.details)
