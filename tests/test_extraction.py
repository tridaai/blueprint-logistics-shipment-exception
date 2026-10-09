"""Extraction node tests — provider SDKs faked, no network.

In the default (mock) mode the node is a pass-through: document fields
are recorded as source-provided with no confidence claimed. With an LLM
backend, fields come from structured extraction over the document text
and deterministic code cross-checks them against the provided values.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.extractor import extract_documents, extraction_discrepancies
from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import MockModelBackend, OpenAIBackend
from shipment_agent.schemas import ShipmentInput

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "RETRIEVER",
]

SHIPMENT_WITH_DOCS = {
    "shipment_id": "EXT-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Retail Co",
    "latest_event": "Delayed at regional hub due to weather hold",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "documents": [
        {
            "doc_type": "bol",
            "document_id": "BOL-EXT-1",
            "raw_text": (
                "BILL OF LADING BOL-EXT-1. Consignee: Synthetic Retail Co. "
                "SKU: SKU-100. Quantity: 120 units. Weight: 840 kg."
            ),
            "fields": {
                "consignee": "Synthetic Retail Co",
                "quantity_units": "120",
                "weight_kg": "840",
                "sku": "SKU-100",
            },
        },
        {
            "doc_type": "invoice",
            "document_id": "INV-EXT-1",
            "raw_text": (
                "COMMERCIAL INVOICE INV-EXT-1. Consignee: Synthetic Retail Co. "
                "SKU: SKU-100. Quantity: 100 units. Weight: 840 kg."
            ),
            "fields": {
                "consignee": "Synthetic Retail Co",
                "quantity_units": "100",
                "weight_kg": "840",
                "sku": "SKU-100",
            },
        },
    ],
}

# What the fake extractor "reads" from the BOL text: quantity 999 is NOT
# in the text or the provided fields — it simulates a model misread so the
# cross-check has a real mismatch to catch.
EXTRACTION_TEXT = (
    '{"quantity_units": {"value": "999", "confidence": 0.61},'
    ' "weight_kg": {"value": "840", "confidence": 0.97},'
    ' "consignee": {"value": "Synthetic Retail Co", "confidence": 0.99},'
    ' "sku": {"value": null, "confidence": 0.0}}'
)

CLEAN_EXTRACTION_TEXT = (
    '{"quantity_units": {"value": "120", "confidence": 0.98},'
    ' "weight_kg": {"value": "840", "confidence": 0.97},'
    ' "consignee": {"value": "Synthetic Retail Co", "confidence": 0.99},'
    ' "sku": {"value": "SKU-100", "confidence": 0.95}}'
)

DRAFT_TEXT = (
    "Subject: Update on shipment EXT-1: delay\n\n"
    "Dear Synthetic Retail Co,\n\n"
    "Your shipment EXT-1 is delayed. [POL-DELAY-01] applies. "
    "The next update will arrive within one business day."
)


class _FakeCompletions:
    def __init__(self, client: "FakeOpenAI") -> None:
        self._client = client

    def create(self, model=None, max_tokens=None, messages=None):
        system = " ".join(messages[0]["content"].split())
        if "extracting structured fields" in system:
            text = self._client.extraction_text
        elif "suggesting an exception classification" in system:
            text = self._client.suggestion_text
        else:
            text = self._client.draft_text
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeOpenAI:
    instances: list["FakeOpenAI"] = []
    draft_text = DRAFT_TEXT
    extraction_text = EXTRACTION_TEXT
    suggestion_text = (
        '{"exception_type": "delay", "severity": "high", "confidence": 0.9,'
        ' "rationale": "Computed delay dominates."}'
    )

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
    FakeOpenAI.extraction_text = EXTRACTION_TEXT
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return FakeOpenAI


def _shipment() -> ShipmentInput:
    return ShipmentInput.model_validate(SHIPMENT_WITH_DOCS)


def test_mock_mode_passes_fields_through_as_source_provided():
    extractions = extract_documents(_shipment(), MockModelBackend())
    assert len(extractions) == 2
    for extraction in extractions:
        assert extraction.source == "provided"
        assert extraction.fields
        for field in extraction.fields:
            assert field.status == "source_provided"
            assert field.confidence is None
            assert field.extracted_value == field.provided_value
    assert extraction_discrepancies(extractions) == []


def test_llm_extraction_cross_check_catches_misread(fake_openai):
    backend = OpenAIBackend()
    extractions = extract_documents(_shipment(), backend)
    bol = extractions[0]
    assert bol.source == "llm"
    by_field = {f.field: f for f in bol.fields}
    # The model "read" 999 where the source system provided 120.
    assert by_field["quantity_units"].status == "mismatch"
    assert by_field["quantity_units"].extracted_value == "999"
    assert by_field["quantity_units"].confidence == 0.61
    assert by_field["weight_kg"].status == "match"
    assert by_field["consignee"].status == "match"
    # sku present in provided fields but the extractor returned null.
    assert by_field["sku"].status == "missing_in_text"
    discrepancies = extraction_discrepancies(extractions)
    assert any("quantity_units" in d and "BOL-EXT-1" in d for d in discrepancies)
    assert any("sku" in d for d in discrepancies)


def test_llm_extraction_clean_match(fake_openai):
    FakeOpenAI.extraction_text = CLEAN_EXTRACTION_TEXT
    backend = OpenAIBackend()
    extractions = extract_documents(_shipment(), backend)
    bol = extractions[0]
    assert all(f.status == "match" for f in bol.fields)
    assert {f.field: f.confidence for f in bol.fields}["quantity_units"] == 0.98


def test_malformed_extraction_falls_back_to_provided(fake_openai):
    FakeOpenAI.extraction_text = "not json at all"
    backend = OpenAIBackend()
    extractions = extract_documents(_shipment(), backend)
    assert all(e.source == "provided" for e in extractions)


def test_extraction_appears_in_result_and_trace_mock_mode():
    result = run_shipment(_shipment(), backend=MockModelBackend())
    assert len(result.extractions) == 2
    extract_step = result.trace[0]
    assert extract_step.name == "extract"
    assert any("source-provided" in d for d in extract_step.details)
    ingest_step = result.trace[1]
    assert any("tool call: compute_delay_hours" in d for d in ingest_step.details)
    assert any("tool call: compare_documents" in d for d in ingest_step.details)


def test_extraction_appears_in_trace_llm_mode(fake_openai):
    FakeOpenAI.extraction_text = CLEAN_EXTRACTION_TEXT
    result = run_shipment(_shipment(), backend=OpenAIBackend())
    extract_step = result.trace[0]
    assert any("tool call: extract_document_fields(BOL-EXT-1)" in d for d in extract_step.details)
    assert result.extractions[0].source == "llm"
