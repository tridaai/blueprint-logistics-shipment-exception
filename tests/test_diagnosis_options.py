"""Diagnose + recovery-options tests.

Default mode: the diagnosis is template-composed from the same evidence
structure, options come from the template proposer — and every number is
computed by the deterministic scorer. Provider mode (fake SDK): the LLM
composes the diagnosis and proposes option kinds; scoring stays in code.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import MockModelBackend, OpenAIBackend
from shipment_agent.options import build_recovery_options, score_option
from shipment_agent.schemas import ShipmentInput

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "RETRIEVER",
]

DELAY_SHIPMENT = {
    "shipment_id": "DIA-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

MISMATCH_SHIPMENT = {
    "shipment_id": "DIA-2",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "latest_event": "Held pending document review",
    "documents": [
        {"doc_type": "bol", "document_id": "BOL-D2", "raw_text": "BOL",
         "fields": {"quantity_units": "120", "weight_kg": "840"}},
        {"doc_type": "invoice", "document_id": "INV-D2", "raw_text": "INV",
         "fields": {"quantity_units": "100", "weight_kg": "840"}},
    ],
}

CLEAN_SHIPMENT = {
    "shipment_id": "DIA-3",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "latest_event": "In transit, on schedule",
    "documents": [],
}

DIAGNOSIS_TEXT = (
    '{"root_cause": "A weather hold at the regional hub consumed the schedule '
    'buffer; the computed delay is 36.0 hours [POL-DELAY-01].",'
    ' "summary": "Carrier weather hold caused a 36-hour delay."}'
)
OPTIONS_TEXT = (
    '[{"kind": "reroute", "title": "Send it via Nashville",'
    '  "description": "Move the shipment through the Nashville hub tonight."},'
    ' {"kind": "expedite", "title": "Upgrade to air",'
    '  "description": "Fly the remaining leg."},'
    ' {"kind": "bogus_kind", "title": "Nope", "description": "Invalid kind."}]'
)
CLASSIFY_TEXT = (
    '{"exception_type": "delay", "severity": "high", "confidence": 0.88,'
    ' "rationale": "Computed delay dominates."}'
)
DRAFT_TEXT = (
    "Subject: Update on shipment DIA-1: delay\n\n"
    "Dear Synthetic Customer,\n\n"
    "Your shipment DIA-1 is delayed. [POL-DELAY-01] applies. "
    "The next update will arrive within one business day."
)


class _FakeCompletions:
    def __init__(self, client: "FakeOpenAI") -> None:
        self._client = client

    def create(self, model=None, max_tokens=None, messages=None):
        system = " ".join(messages[0]["content"].split())
        client = self._client
        if "extracting structured fields" in system:
            text = client.extraction_text
        elif "classifying a shipment exception" in system:
            text = CLASSIFY_TEXT
        elif "diagnosing the root cause" in system:
            text = client.diagnosis_text
        elif "proposing recovery options" in system:
            text = client.options_text
        else:
            text = DRAFT_TEXT
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeOpenAI:
    instances: list["FakeOpenAI"] = []
    diagnosis_text = DIAGNOSIS_TEXT
    options_text = OPTIONS_TEXT
    extraction_text = "{}"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))
        FakeOpenAI.instances.append(self)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.model_backends.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.retriever.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAI.instances = []
    FakeOpenAI.diagnosis_text = DIAGNOSIS_TEXT
    FakeOpenAI.options_text = OPTIONS_TEXT
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return FakeOpenAI


def _run(shipment: dict, backend) -> object:
    return run_shipment(ShipmentInput.model_validate(shipment), backend=backend)


# --------------------------------------------------------------------------
# Deterministic scorer
# --------------------------------------------------------------------------

def test_scorer_is_deterministic_and_recommends_expedite_for_big_delay():
    eta, cost, sla, score = score_option("expedite", 36.0, "high")
    assert (eta, cost, sla, score) == score_option("expedite", 36.0, "high")
    options = build_recovery_options(
        exception_type="delay", severity="high", delay_hours=36.0
    )
    recommended = next(o for o in options if o.recommended)
    assert recommended.kind == "expedite"
    assert sum(o.recommended for o in options) == 1
    # Scores come from the scorer, not from the proposer.
    for option in options:
        assert option.score == score_option(option.kind, 36.0, "high")[3]


def test_scorer_recommends_document_correction_for_mismatch():
    options = build_recovery_options(
        exception_type="document_mismatch", severity="medium", delay_hours=None
    )
    recommended = next(o for o in options if o.recommended)
    assert recommended.kind == "correct_documents"
    assert [o.kind for o in options] == [
        "correct_documents", "partial_reship", "wait_and_monitor",
    ]


def test_none_exception_gets_single_continue_option():
    options = build_recovery_options(
        exception_type="none", severity="low", delay_hours=None
    )
    assert len(options) == 1
    assert options[0].kind == "wait_and_monitor"
    assert options[0].recommended is True


# --------------------------------------------------------------------------
# Default (mock) mode through the graph
# --------------------------------------------------------------------------

def test_mock_diagnosis_is_template_with_cited_evidence():
    result = _run(DELAY_SHIPMENT, MockModelBackend())
    diagnosis = result.diagnosis
    assert diagnosis is not None
    assert diagnosis.source == "template"
    assert "36.0" in diagnosis.root_cause
    assert any("compute_delay_hours" in e for e in diagnosis.evidence)
    assert diagnosis.citations == [p.policy_id for p in result.policies]
    diagnose_step = next(s for s in result.trace if s.name == "diagnose")
    assert diagnosis.summary in diagnose_step.summary


def test_mock_options_scored_and_grounded_in_draft_and_packet():
    result = _run(DELAY_SHIPMENT, MockModelBackend())
    assert len(result.recovery_options) == 3
    recommended = next(o for o in result.recovery_options if o.recommended)
    assert result.recommended_option_id == recommended.option_id
    # The draft grounds on the recommended option.
    assert recommended.title in result.draft.body
    # The claim packet carries the diagnosis + scored options.
    packet = result.draft.claim_packet
    assert packet["recommended_option_id"] == recommended.option_id
    assert packet["recovery_options"][0]["option_id"] == "OPT-1"
    assert "root_cause" in packet["diagnosis"]


def test_mock_mismatch_diagnosis_names_conflicting_fields():
    result = _run(MISMATCH_SHIPMENT, MockModelBackend())
    assert result.classification.exception_type.value == "document_mismatch"
    assert "quantity_units" in result.diagnosis.root_cause
    assert any("compare_documents" in e for e in result.diagnosis.evidence)


def test_options_trace_step_shows_scores():
    result = _run(DELAY_SHIPMENT, MockModelBackend())
    options_step = next(s for s in result.trace if s.name == "options")
    assert any("<- recommended" in d for d in options_step.details)
    assert all("score" in d for d in options_step.details)


# --------------------------------------------------------------------------
# Provider mode (fake SDK)
# --------------------------------------------------------------------------

def test_llm_diagnosis_and_options_with_code_scoring(fake_openai):
    result = _run(DELAY_SHIPMENT, OpenAIBackend())
    assert result.diagnosis.source == "llm"
    assert "weather hold" in result.diagnosis.root_cause.lower()
    # Evidence stays deterministic even when the prose is the model's.
    assert any("compute_delay_hours" in e for e in result.diagnosis.evidence)
    kinds = [o.kind for o in result.recovery_options]
    assert kinds == ["reroute", "expedite"]  # bogus kind was filtered out
    for option in result.recovery_options:
        assert option.score == score_option(option.kind, 36.0, "high")[3]
    recommended = next(o for o in result.recovery_options if o.recommended)
    assert recommended.kind == "expedite"  # highest deterministic score


def test_llm_options_fall_back_to_template_when_unusable(fake_openai):
    FakeOpenAI.options_text = "no json here"
    result = _run(DELAY_SHIPMENT, OpenAIBackend())
    assert [o.kind for o in result.recovery_options] == [
        "expedite", "reroute", "wait_and_monitor",
    ]


def test_llm_diagnosis_falls_back_to_template_when_unusable(fake_openai):
    FakeOpenAI.diagnosis_text = "not json"
    result = _run(DELAY_SHIPMENT, OpenAIBackend())
    assert result.diagnosis.source == "template"
