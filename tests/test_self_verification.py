"""Self-verification tests — the agent critiques its own draft.

Provider mode: an LLM critique call (faked SDK) returns a structured
verdict. Default mode: a deterministic evidence checklist produces the
same verdict shape, honestly labelled. Either way the verdict lands in
the result, the claim packet, and the trace between draft and validate.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import OpenAIBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import load_sample_shipments
from shipment_agent.schemas import DraftOutput, ShipmentInput
from shipment_agent.verify import checklist_verification, verify_draft

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "LLM_TIMEOUT_SECONDS", "RETRIEVER",
]

CLASSIFY_TEXT = (
    '{"exception_type": "delay", "severity": "high", "confidence": 0.88,'
    ' "rationale": "Computed delay dominates."}'
)
DIAGNOSIS_TEXT = (
    '{"root_cause": "Weather hold at the hub.", "summary": "Carrier weather hold."}'
)
OPTIONS_TEXT = '[{"kind": "reroute", "title": "Via Nashville", "description": "Reroute tonight."}]'
VERIFY_TEXT = (
    '{"grounded": false, "issues": ["Draft commits to a phone call within one '
    'hour; no verified fact supports that commitment."], "summary": "One '
    'unsupported commitment found."}'
)
DRAFT_TEXT = (
    "Subject: Update on shipment VER-1: delay\n\n"
    "Dear Synthetic Customer,\n\n"
    "Your shipment VER-1 is delayed by 36 hours. [POL-DELAY-01] applies. "
    "We will call you within one hour."
)

SHIPMENT = {
    "shipment_id": "VER-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


class APIConnectionError(Exception):
    pass


class _FakeCompletions:
    def __init__(self, client: "FakeOpenAI") -> None:
        self._client = client

    def create(self, model=None, max_tokens=None, messages=None):
        system = " ".join(messages[0]["content"].split())
        if "extracting structured fields" in system:
            kind, text = "extract", "{}"
        elif "classifying a shipment exception" in system:
            kind, text = "classify", CLASSIFY_TEXT
        elif "diagnosing the root cause" in system:
            kind, text = "diagnose", DIAGNOSIS_TEXT
        elif "proposing recovery options" in system:
            kind, text = "options", OPTIONS_TEXT
        elif "verifying whether a drafted customer update" in system:
            kind, text = "verify", VERIFY_TEXT
        else:
            kind, text = "draft", DRAFT_TEXT
        if kind in self._client.fail_kinds:
            raise APIConnectionError("Connection error.")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeOpenAI:
    instances: list["FakeOpenAI"] = []
    fail_kinds: set = set()

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


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAI.instances = []
    FakeOpenAI.fail_kinds = set()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return FakeOpenAI


def _sample(shipment_id: str) -> ShipmentInput:
    return ShipmentInput.model_validate(
        next(s for s in load_sample_shipments() if s["shipment_id"] == shipment_id)
    )


# --------------------------------------------------------------------------
# Default mode — deterministic checklist, honestly labelled
# --------------------------------------------------------------------------

def test_mock_run_verifies_with_the_checklist():
    result = run_shipment(_sample("SYN-1001"))
    assert result.verification is not None
    assert result.verification.source == "checklist"
    assert result.verification.grounded is True
    assert result.verification.issues == []
    assert result.draft.claim_packet["verification"]["source"] == "checklist"


def test_verify_step_sits_between_draft_and_validate_in_the_trace():
    result = run_shipment(_sample("SYN-1001"))
    names = [s.name for s in result.trace]
    assert names == [
        "extract", "ingest", "classify", "retrieve", "diagnose",
        "options", "draft", "verify", "validate", "human_approval",
    ]
    verify_step = next(s for s in result.trace if s.name == "verify")
    assert verify_step.status == "passed"
    assert any("grounded" in d for d in verify_step.details)


# --------------------------------------------------------------------------
# The checklist itself — each check compares against a computed fact
# --------------------------------------------------------------------------

POLICIES = [{"policy_id": "POL-DELAY-01", "title": "Delay policy", "snippet": "..."}]


def _draft(body: str) -> DraftOutput:
    return DraftOutput(subject="Update on shipment SYN-1001", body=body, citations=["POL-DELAY-01"])


def test_checklist_flags_wrong_delay_figure():
    verdict = checklist_verification(
        draft=_draft("Your shipment is delayed by 99 hours. [POL-DELAY-01]"),
        delay_hours=36.0, policies=POLICIES, shipment_id="SYN-1001",
    )
    assert verdict.grounded is False
    assert any("99" in i and "36" in i for i in verdict.issues)


def test_checklist_flags_unretrieved_policy_citation():
    verdict = checklist_verification(
        draft=_draft("Per the policy [POL-FAKE-99] we escalate."),
        delay_hours=None, policies=POLICIES, shipment_id="SYN-1001",
    )
    assert verdict.grounded is False
    assert any("POL-FAKE-99" in i for i in verdict.issues)


def test_checklist_flags_foreign_shipment_reference_and_money():
    verdict = checklist_verification(
        draft=_draft("As with shipment SYN-9999, a $500 credit applies."),
        delay_hours=None, policies=POLICIES, shipment_id="SYN-1001",
    )
    assert verdict.grounded is False
    assert any("SYN-9999" in i for i in verdict.issues)
    assert any("monetary" in i for i in verdict.issues)


# --------------------------------------------------------------------------
# Provider mode — LLM critique, same verdict shape
# --------------------------------------------------------------------------

def test_llm_critique_verdict_flows_to_result_packet_and_trace(fake_openai):
    result = run_shipment(
        ShipmentInput.model_validate(SHIPMENT),
        backend=OpenAIBackend(),
        retriever=KeywordRetriever(),
    )
    assert result.verification is not None
    assert result.verification.source == "llm"
    assert result.verification.grounded is False
    assert any("phone call" in i for i in result.verification.issues)
    assert result.draft.claim_packet["verification"]["grounded"] is False
    verify_step = next(s for s in result.trace if s.name == "verify")
    assert verify_step.status == "failed"
    assert any("issue:" in d for d in verify_step.details)


def test_llm_critique_failure_degrades_to_checklist_with_note(fake_openai):
    FakeOpenAI.fail_kinds = {"verify"}
    verdict = verify_draft(
        draft=_draft("Your shipment is delayed by 36 hours. [POL-DELAY-01]"),
        classification={"exception_type": "delay", "severity": "high"},
        delay_hours=36.0,
        mismatches=[],
        policies=POLICIES,
        shipment_id="SYN-1001",
        backend=OpenAIBackend(),
    )
    assert verdict.source == "checklist"
    assert verdict.grounded is True
    assert "LLM verification failed" in verdict.note
    assert "openai provider call failed" in verdict.note
