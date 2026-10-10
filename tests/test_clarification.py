"""Clarification-request tests (the information-needed flow).

Trigger (all deterministic): the final classification is ``none``
below 0.6 confidence AND concrete inputs are missing — the BOL/invoice
pair (document check skipped) or a critical field the extraction
could not find in the document text. The pipeline then attaches a
composed request listing exactly what is missing; the case still
stops at the human-approval gate and nothing is sent.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import shipment_agent.graph as graph_module
from shipment_agent.clarify import build_information_request, missing_items
from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import MockModelBackend, OpenAIBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.schemas import Classification, ExceptionType, Severity, ShipmentInput
from shipment_agent.tools import document_pair_warning

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "RETRIEVER",
]

UNDER_DETERMINED = {
    "shipment_id": "CLR-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Acme Parts",
    "latest_event": "In transit",
    "documents": [],
}

NONE_LOW = {"exception_type": "none", "severity": "low", "confidence": 0.5}


def _shipment(payload=UNDER_DETERMINED) -> ShipmentInput:
    return ShipmentInput.model_validate(payload)


def _request(classification, warning, extractions=None, backend=None, shipment=None):
    return build_information_request(
        shipment=shipment or _shipment(),
        classification=classification,
        document_check_warning=warning,
        extractions=extractions or [],
        backend=backend or MockModelBackend(),
    )


# --------------------------------------------------------------------------
# Trigger matrix (deterministic)
# --------------------------------------------------------------------------

def test_triggers_when_none_low_confidence_and_pair_missing():
    shipment = _shipment()
    request = _request(NONE_LOW, document_pair_warning(shipment.documents), shipment=shipment)
    assert request is not None
    assert request.source == "template"  # mock mode composes from the template
    assert any("bill of lading" in item for item in request.missing_items)
    assert any("invoice" in item for item in request.missing_items)
    assert "CLR-1" in request.message
    for item in request.missing_items:
        assert item in request.message  # the message lists exactly the items


def test_no_trigger_when_confident_none():
    shipment = _shipment()
    confident = {"exception_type": "none", "severity": "low", "confidence": 0.9}
    assert _request(confident, document_pair_warning(shipment.documents)) is None


def test_no_trigger_for_a_real_exception_even_at_low_confidence():
    shipment = _shipment()
    delay = {"exception_type": "delay", "severity": "low", "confidence": 0.5}
    assert _request(delay, document_pair_warning(shipment.documents)) is None


def test_no_trigger_when_nothing_is_missing():
    # Pair present, no extraction gaps: a low-confidence "none" alone
    # is not enough — there must be something concrete to ask for.
    payload = {
        **UNDER_DETERMINED,
        "documents": [
            {"doc_type": "bol", "document_id": "BOL-C1", "fields": {}},
            {"doc_type": "invoice", "document_id": "INV-C1", "fields": {}},
        ],
    }
    shipment = _shipment(payload)
    assert document_pair_warning(shipment.documents) is None
    assert _request(NONE_LOW, None, shipment=shipment) is None


def test_triggers_on_a_critical_field_missing_in_text():
    extractions = [
        {
            "document_id": "BOL-C2",
            "fields": [
                {"field": "quantity_units", "status": "missing_in_text"},
                {"field": "weight_kg", "status": "match"},
            ],
        }
    ]
    request = _request(NONE_LOW, None, extractions=extractions)
    assert request is not None
    assert any("quantity_units" in item for item in request.missing_items)
    assert not any("weight_kg" in item for item in request.missing_items)


def test_missing_items_names_only_the_absent_document():
    payload = {
        **UNDER_DETERMINED,
        "documents": [{"doc_type": "bol", "document_id": "BOL-C3", "fields": {}}],
    }
    shipment = _shipment(payload)
    items = missing_items(shipment, document_pair_warning(shipment.documents), [])
    assert any("invoice" in item for item in items)
    assert not any("bill of lading" in item for item in items)


# --------------------------------------------------------------------------
# Provider-mode composition (fake SDK)
# --------------------------------------------------------------------------

INFO_MESSAGE = (
    "Subject: Information needed — shipment CLR-1\n\n"
    "Hello Synthetic Carrier operations team, please send the bill of "
    "lading and the commercial invoice for shipment CLR-1."
)


class _InfoCompletions:
    def create(self, model=None, max_tokens=None, messages=None, tools=None):
        system = " ".join(messages[0]["content"].split())
        if "operations coordinator" in system:
            text = INFO_MESSAGE
        elif "classifying a shipment exception" in system:
            text = '{"exception_type": "none", "severity": "low", "confidence": 0.5, "rationale": "x"}'
        elif "diagnosing the root cause" in system:
            text = '{"root_cause": "Undetermined.", "summary": "Insufficient data."}'
        elif "proposing recovery options" in system:
            text = '[{"kind": "wait_and_monitor", "title": "Hold", "description": "Wait."}]'
        elif "verifying whether a drafted" in system:
            text = '{"grounded": true, "issues": [], "summary": "Grounded."}'
        else:
            text = (
                "Subject: Status update on shipment CLR-1\n\nDear Acme Parts,\n\n"
                "Your shipment CLR-1 is in transit. The next update will follow."
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class InfoOpenAI:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=_InfoCompletions())


@pytest.fixture
def fake_info_openai(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=InfoOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("shipment_agent.model_backends.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.retriever.load_dotenv", lambda *a, **k: None)
    return InfoOpenAI


def test_provider_mode_composes_with_the_llm(fake_info_openai):
    shipment = _shipment()
    request = _request(
        NONE_LOW,
        document_pair_warning(shipment.documents),
        backend=OpenAIBackend(),
        shipment=shipment,
    )
    assert request is not None
    assert request.source == "llm"
    assert request.message == INFO_MESSAGE
    # The missing-items list still comes from code, not the model.
    assert any("bill of lading" in item for item in request.missing_items)


# --------------------------------------------------------------------------
# Graph wiring
# --------------------------------------------------------------------------

def test_graph_attaches_the_request_and_still_stops_at_the_gate(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "classify_shipment",
        lambda shipment, mismatches: Classification(
            exception_type=ExceptionType.NONE,
            severity=Severity.LOW,
            confidence=0.5,
            signals=[],
            rationale="Nothing determinable from the data on file.",
        ),
    )
    result = run_shipment(_shipment(), backend=MockModelBackend(), retriever=KeywordRetriever())
    assert result.classification.exception_type == ExceptionType.NONE
    assert result.needs_information is True
    assert result.information_request is not None
    assert result.information_request.missing_items  # lists what is missing
    assert result.approval_status == "awaiting_approval"  # gate unchanged
    assert result.external_action_taken is False  # the request was NOT sent
    assert result.draft.claim_packet["information_request"]["missing_items"]
    gate = next(s for s in result.trace if s.name == "human_approval")
    assert any("information requested" in d for d in gate.details)


def test_graph_normal_case_needs_no_information():
    payload = {
        "shipment_id": "CLR-9",
        "origin": "Memphis, TN",
        "destination": "Charlotte, NC",
        "scheduled_delivery": "2026-10-10T09:00:00",
        "estimated_delivery": "2026-10-11T21:00:00",
        "latest_event": "Delayed at regional hub due to weather hold",
        "documents": [],
    }
    result = run_shipment(
        ShipmentInput.model_validate(payload),
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
    )
    assert result.classification.exception_type == ExceptionType.DELAY
    assert result.needs_information is False
    assert result.information_request is None
    assert "information_request" not in result.draft.claim_packet
