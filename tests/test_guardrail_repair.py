"""Bounded repair loop tests.

On guardrail failure the agent may redraft ONCE (configurable) with the
failure reasons + self-verification issues fed back, then re-validate.
The original failure is preserved and the attempt is flagged. With
GUARDRAIL_REPAIR=off the failing sample behaves exactly as a no-repair
pipeline: draft fails, approval is refused. The guardrail rules
themselves never change.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import OpenAIBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import load_sample_shipments
from shipment_agent.schemas import ShipmentInput
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "LLM_TIMEOUT_SECONDS", "RETRIEVER", "GUARDRAIL_REPAIR",
    "GUARDRAIL_REPAIR_MAX_ATTEMPTS",
]

CLASSIFY_TEXT = (
    '{"exception_type": "delay", "severity": "high", "confidence": 0.88,'
    ' "rationale": "Computed delay dominates."}'
)
DIAGNOSIS_TEXT = (
    '{"root_cause": "Weather hold at the hub.", "summary": "Carrier weather hold."}'
)
OPTIONS_TEXT = '[{"kind": "reroute", "title": "Via Nashville", "description": "Reroute tonight."}]'
VERIFY_TEXT = '{"grounded": true, "issues": [], "summary": "Grounded."}'
BAD_DRAFT = (
    "Subject: Update on shipment REP-1\n\n"
    "Dear Synthetic Customer,\n\n"
    "Your shipment REP-1 is delayed. We guarantee a full refund for the "
    "delay. [POL-DELAY-01] applies."
)
CLEAN_DRAFT = (
    "Subject: Update on shipment REP-1: delay\n\n"
    "Dear Synthetic Customer,\n\n"
    "Your shipment REP-1 is delayed. [POL-DELAY-01] applies. "
    "The next update will arrive within one business day."
)

SHIPMENT = {
    "shipment_id": "REP-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


class _FakeCompletions:
    def __init__(self, client: "FakeOpenAI") -> None:
        self._client = client

    def create(self, model=None, max_tokens=None, messages=None):
        system = " ".join(messages[0]["content"].split())
        user = messages[-1]["content"]
        if "extracting structured fields" in system:
            text = "{}"
        elif "classifying a shipment exception" in system:
            text = CLASSIFY_TEXT
        elif "diagnosing the root cause" in system:
            text = DIAGNOSIS_TEXT
        elif "proposing recovery options" in system:
            text = OPTIONS_TEXT
        elif "verifying whether a drafted customer update" in system:
            text = VERIFY_TEXT
        elif "CORRECTION REQUIRED" in user:
            # The repair redraft: feedback-aware, clean unless the client
            # is configured to keep failing.
            text = BAD_DRAFT if self._client.always_bad else CLEAN_DRAFT
        else:
            text = BAD_DRAFT
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeOpenAI:
    instances: list["FakeOpenAI"] = []
    always_bad: bool = False

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
    monkeypatch.setattr("shipment_agent.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.graph.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAI.instances = []
    FakeOpenAI.always_bad = False
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return FakeOpenAI


def _sample(shipment_id: str) -> dict:
    return next(s for s in load_sample_shipments() if s["shipment_id"] == shipment_id)


# --------------------------------------------------------------------------
# The guardrail-failure sample (SYN-1013), repair ON (default) vs OFF
# --------------------------------------------------------------------------

def test_syn1013_default_mode_attempts_repair_but_still_fails():
    """Mock drafting is deterministic, so the redraft fails identically —
    the attempt is surfaced, the original failure preserved, and the
    approval gate still refuses."""
    result = run_shipment(ShipmentInput.model_validate(_sample("SYN-1013")))
    assert result.validation.passed is False
    assert result.repair_attempted is True
    assert result.repair_attempts == 1
    assert result.repaired is False
    assert result.original_validation is not None
    assert result.original_validation.errors == result.validation.errors
    assert any("we guarantee" in e for e in result.validation.errors)
    validate_step = next(s for s in result.trace if s.name == "validate")
    assert any(d.startswith("repair:") for d in validate_step.details)
    assert any(d.startswith("original failure:") for d in validate_step.details)


def test_syn1013_repair_off_behaves_exactly_as_before(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_REPAIR", "off")
    result = run_shipment(ShipmentInput.model_validate(_sample("SYN-1013")))
    assert result.validation.passed is False
    assert result.repair_attempted is False
    assert result.repair_attempts == 0
    assert result.repaired is False
    assert result.original_validation is None


def test_syn1013_approval_still_refused_after_failed_repair():
    service = ShipmentService(store=InMemoryStore())
    service.analyze(_sample("SYN-1013"))
    with pytest.raises(ValueError, match="guardrail"):
        service.approve("SYN-1013", approver="ops-lead")


def test_passing_draft_never_triggers_repair():
    result = run_shipment(ShipmentInput.model_validate(_sample("SYN-1001")))
    assert result.validation.passed is True
    assert result.repair_attempted is False
    assert result.original_validation is None


# --------------------------------------------------------------------------
# Provider mode — a feedback-aware redraft can actually repair
# --------------------------------------------------------------------------

def test_llm_repair_redrafts_with_feedback_and_passes(fake_openai):
    result = run_shipment(
        ShipmentInput.model_validate(SHIPMENT),
        backend=OpenAIBackend(),
        retriever=KeywordRetriever(),
    )
    assert result.repair_attempted is True
    assert result.repaired is True
    assert result.validation.passed is True
    assert "guarantee" not in result.draft.body.lower()
    assert result.original_validation is not None
    assert any("we guarantee" in e for e in result.original_validation.errors)
    # The repaired draft was re-verified and carries the fresh verdict.
    assert result.draft.claim_packet["verification"]["grounded"] is True


def test_repair_attempts_are_bounded_by_configuration(fake_openai, monkeypatch):
    FakeOpenAI.always_bad = True
    monkeypatch.setenv("GUARDRAIL_REPAIR_MAX_ATTEMPTS", "2")
    result = run_shipment(
        ShipmentInput.model_validate(SHIPMENT),
        backend=OpenAIBackend(),
        retriever=KeywordRetriever(),
    )
    assert result.repair_attempts == 2
    assert result.repaired is False
    assert result.validation.passed is False
