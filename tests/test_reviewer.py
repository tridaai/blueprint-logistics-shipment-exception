"""Independent reviewer tests — the generator/critic split.

Self-verification is the pipeline critiquing itself; the reviewer is a
distinct step after it: its own persona prompt (and optionally its own
model) in provider mode, a deterministic second checklist — different
checks: next-update commitment, claim-packet completeness, citation
coverage of the diagnosis — in the default mode. A ``block`` verdict
flags the result and disqualifies auto-approval; it never rejects
anything itself. ``REVIEWER=off`` disables the step.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.autonomy import compute_autonomy
from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import OpenAIBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.reviewer import checklist_review
from shipment_agent.samples import load_sample_shipments
from shipment_agent.schemas import DraftOutput, ShipmentInput

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "LLM_TIMEOUT_SECONDS", "RETRIEVER", "REVIEWER", "REVIEWER_MODEL",
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
REVIEW_BLOCK_TEXT = (
    '{"verdict": "block", "findings": ["The draft promises a callback the '
    'facts do not support; the approver cannot clear this as written."]}'
)
REVIEW_PASS_TEXT = '{"verdict": "pass", "findings": []}'
DRAFT_TEXT = (
    "Subject: Update on shipment REV-1: delay\n\n"
    "Dear Synthetic Customer,\n\n"
    "Your shipment REV-1 is delayed. [POL-DELAY-01] applies. "
    "The next update will arrive within one business day."
)

SHIPMENT = {
    "shipment_id": "REV-1",
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
        elif "independent operations reviewer" in system:
            kind, text = "review", self._client.review_text
        else:
            kind, text = "draft", DRAFT_TEXT
        if kind in self._client.fail_kinds:
            raise APIConnectionError("Connection error.")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeOpenAI:
    instances: list["FakeOpenAI"] = []
    fail_kinds: set = set()
    review_text: str = REVIEW_BLOCK_TEXT

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
    monkeypatch.setattr("shipment_agent.reviewer.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.config.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAI.instances = []
    FakeOpenAI.fail_kinds = set()
    FakeOpenAI.review_text = REVIEW_BLOCK_TEXT
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return FakeOpenAI


def _sample(shipment_id: str) -> ShipmentInput:
    return ShipmentInput.model_validate(
        next(s for s in load_sample_shipments() if s["shipment_id"] == shipment_id)
    )


def _run_provider():
    return run_shipment(
        ShipmentInput.model_validate(SHIPMENT),
        backend=OpenAIBackend(),
        retriever=KeywordRetriever(),
    )


# --------------------------------------------------------------------------
# Default mode — deterministic second checklist, honestly labelled
# --------------------------------------------------------------------------

def test_mock_review_is_a_checklist_and_passes_a_clean_sample():
    result = run_shipment(_sample("SYN-1001"))
    assert result.review is not None
    assert result.review.source == "checklist"
    assert result.review.verdict == "pass"
    assert result.review.findings == []
    assert result.reviewer_blocked is False
    assert result.draft.claim_packet["review"]["verdict"] == "pass"


def test_review_step_sits_between_verify_and_validate_in_the_trace():
    result = run_shipment(_sample("SYN-1001"))
    names = [s.name for s in result.trace]
    assert names.index("verify") < names.index("review") < names.index("validate")
    review_step = next(s for s in result.trace if s.name == "review")
    assert review_step.status == "passed"
    assert any("verdict: pass" in d for d in review_step.details)


def _packet(**overrides) -> dict:
    packet = {
        "diagnosis": {"root_cause": "Weather hold.", "citations": ["POL-DELAY-01"]},
        "policy_citations": ["POL-DELAY-01"],
        "recovery_options": [{"option_id": "OPT-1"}],
        "recommended_option_id": "OPT-1",
    }
    packet.update(overrides)
    return packet


def test_checklist_blocks_a_packet_with_no_diagnosis():
    draft = DraftOutput(
        subject="Update",
        body="Your shipment is delayed. The next update will arrive within one business day.",
        claim_packet=_packet(diagnosis=None),
        citations=["POL-DELAY-01"],
    )
    verdict = checklist_review(
        draft=draft,
        classification={"exception_type": "delay", "severity": "high"},
        diagnosis={"citations": ["POL-DELAY-01"]},
    )
    assert verdict.verdict == "block"
    assert any("no diagnosis" in f for f in verdict.findings)


def test_checklist_flags_missing_next_update_and_uncovered_citation():
    draft = DraftOutput(
        subject="Update",
        body="Your shipment is delayed and we are working on it.",
        claim_packet=_packet(recommended_option_id=None),
        citations=["POL-DELAY-01"],
    )
    verdict = checklist_review(
        draft=draft,
        classification={"exception_type": "delay", "severity": "high"},
        diagnosis={"citations": ["POL-DELAY-01", "POL-COMM-01"]},
    )
    assert verdict.verdict == "concerns"
    assert any("next update" in f for f in verdict.findings)
    assert any("no recommended recovery option" in f for f in verdict.findings)
    assert any("POL-COMM-01" in f and "does not cite" in f for f in verdict.findings)


# --------------------------------------------------------------------------
# Provider mode — the LLM reviewer, its own persona and verdict
# --------------------------------------------------------------------------

def test_llm_reviewer_block_flags_the_result_and_disqualifies_autonomy(fake_openai):
    result = _run_provider()
    assert result.review is not None
    assert result.review.source == "llm"
    assert result.review.verdict == "block"
    assert any("callback" in f for f in result.review.findings)
    assert result.review.model == "gpt-4o-mini"  # defaults to the run's model
    assert result.reviewer_blocked is True
    assert result.draft.claim_packet["review"]["verdict"] == "block"
    # The block does NOT reject or decide anything by itself.
    assert result.approval_status == "awaiting_approval"
    assert result.autonomy is not None
    assert result.autonomy.eligible_for_auto_approval is False
    assert any("reviewer blocked" in r for r in result.autonomy.reasons)
    review_step = next(s for s in result.trace if s.name == "review")
    assert review_step.status == "failed"


def test_llm_reviewer_pass_leaves_the_result_unflagged(fake_openai):
    FakeOpenAI.review_text = REVIEW_PASS_TEXT
    result = _run_provider()
    assert result.review is not None
    assert result.review.verdict == "pass"
    assert result.reviewer_blocked is False


def test_reviewer_model_env_overrides_the_run_model(fake_openai, monkeypatch):
    monkeypatch.setenv("REVIEWER_MODEL", "gpt-4o")
    result = _run_provider()
    assert result.review is not None
    assert result.review.model == "gpt-4o"


def test_llm_review_failure_degrades_to_checklist_with_note(fake_openai):
    FakeOpenAI.fail_kinds = {"review"}
    result = _run_provider()
    assert result.review is not None
    assert result.review.source == "checklist"
    assert "LLM review failed" in result.review.note
    assert "openai provider call failed" in result.review.note


# --------------------------------------------------------------------------
# Switch + autonomy policy
# --------------------------------------------------------------------------

def test_reviewer_off_disables_the_step(monkeypatch):
    monkeypatch.setenv("REVIEWER", "off")
    result = run_shipment(_sample("SYN-1001"))
    assert result.review is None
    assert result.reviewer_blocked is False
    assert "review" not in result.draft.claim_packet
    review_step = next(s for s in result.trace if s.name == "review")
    assert any("REVIEWER=off" in d for d in review_step.details)


def test_autonomy_policy_treats_a_reviewer_block_as_a_disqualifier():
    rec = compute_autonomy(
        classification={"exception_type": "none", "severity": "low"},
        validation={"passed": True},
        cross_check=None,
        repair_attempted=False,
        reviewer_blocked=True,
    )
    assert rec.eligible_for_auto_approval is False
    assert any("reviewer blocked" in r for r in rec.reasons)
