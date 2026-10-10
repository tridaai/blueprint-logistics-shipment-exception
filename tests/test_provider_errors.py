"""Provider error translation tests — provider SDKs are faked.

The customer-simulation finding: a dead provider (Ollama not running, a
bad base URL) surfaced as a raw SDK traceback after ~38s of silent SDK
retries, and node-level fallbacks degraded silently. Now: SDK retries
are off, transport failures are translated into the product's own
`error: …` style naming the backend, the endpoint, and the likely fix,
and every degradation is recorded in the trace.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import shipment_agent.api as api_module
from shipment_agent import cli
from shipment_agent.diagnosis import build_diagnosis
from shipment_agent.errors import ProviderError
from shipment_agent.extractor import extract_documents
from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import OllamaBackend, OpenAIBackend
from shipment_agent.retriever import KeywordRetriever, get_retriever
from shipment_agent.schemas import ShipmentInput
from shipment_agent.service import ShipmentService

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_BASE_URL",
    "OLLAMA_BASE_URL", "OLLAMA_MODEL", "LLM_TIMEOUT_SECONDS", "RETRIEVER",
]


class APIConnectionError(Exception):
    """Stands in for openai.APIConnectionError (same failure shape)."""


CLASSIFY_TEXT = (
    '{"exception_type": "delay", "severity": "high", "confidence": 0.88,'
    ' "rationale": "Computed delay dominates."}'
)
DIAGNOSIS_TEXT = (
    '{"root_cause": "Weather hold at the hub.", "summary": "Carrier weather hold."}'
)
OPTIONS_TEXT = (
    '[{"kind": "reroute", "title": "Via Nashville", "description": "Reroute tonight."},'
    ' {"kind": "expedite", "title": "Upgrade", "description": "Fly the last leg."}]'
)
DRAFT_TEXT = (
    "Subject: Update on shipment ERR-1: delay\n\n"
    "Dear Synthetic Customer,\n\n"
    "Your shipment ERR-1 is delayed. [POL-DELAY-01] applies. "
    "The next update will arrive within one business day."
)

SHIPMENT = {
    "shipment_id": "ERR-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [
        {
            "doc_type": "bol",
            "document_id": "BOL-E1",
            "raw_text": "Bill of lading. Quantity: 120 units. Weight: 840 kg.",
            "fields": {"quantity_units": "120", "weight_kg": "840"},
        }
    ],
}


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


def _shipment() -> ShipmentInput:
    return ShipmentInput.model_validate(SHIPMENT)


# --------------------------------------------------------------------------
# Retry posture
# --------------------------------------------------------------------------

def test_clients_disable_silent_sdk_retries(fake_openai):
    OpenAIBackend()
    assert FakeOpenAI.instances[-1].kwargs["max_retries"] == 0
    OllamaBackend()
    assert FakeOpenAI.instances[-1].kwargs["max_retries"] == 0


# --------------------------------------------------------------------------
# Translation at the drafting boundary (cannot degrade — clean error)
# --------------------------------------------------------------------------

def test_dead_ollama_surfaces_a_clean_actionable_error(fake_openai):
    FakeOpenAI.fail_kinds = {"draft"}
    backend = OllamaBackend()
    with pytest.raises(ProviderError) as excinfo:
        run_shipment(_shipment(), backend=backend, retriever=KeywordRetriever())
    message = str(excinfo.value)
    assert "ollama provider call failed (connection)" in message
    assert "http://localhost:11434/v1" in message
    assert "ollama serve" in message
    assert "Traceback" not in message


def test_dead_openai_error_names_backend_endpoint_and_fix(fake_openai):
    FakeOpenAI.fail_kinds = {"draft"}
    backend = OpenAIBackend()
    with pytest.raises(ProviderError) as excinfo:
        run_shipment(_shipment(), backend=backend, retriever=KeywordRetriever())
    message = str(excinfo.value)
    assert "openai provider call failed (connection)" in message
    assert "https://api.openai.com/v1" in message
    assert "OPENAI_BASE_URL" in message


def test_api_returns_clean_502_not_a_stack_dump(fake_openai, monkeypatch):
    FakeOpenAI.fail_kinds = {"draft"}
    service = ShipmentService(backend=OpenAIBackend(), retriever=KeywordRetriever())
    monkeypatch.setattr(api_module, "service", service)
    client = TestClient(api_module.app, raise_server_exceptions=False)
    response = client.post("/shipments/analyze", json=SHIPMENT)
    assert response.status_code == 502
    assert "provider call failed" in response.json()["detail"]


def test_cli_exits_with_clean_error_line(fake_openai, monkeypatch, capsys):
    FakeOpenAI.fail_kinds = {"draft"}
    monkeypatch.setenv("MODEL_BACKEND", "ollama")
    monkeypatch.setattr("sys.argv", ["shipment-agent", "--index", "0"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ollama provider call failed")
    assert "Traceback" not in err


# --------------------------------------------------------------------------
# Degradation is consistent AND visible
# --------------------------------------------------------------------------

def test_classify_failure_degrades_to_rules_only_and_is_traced(fake_openai):
    FakeOpenAI.fail_kinds = {"classify"}
    result = run_shipment(_shipment(), backend=OpenAIBackend(), retriever=KeywordRetriever())
    assert result.cross_check is not None
    assert result.cross_check.resolution == "rules_only"
    classify_step = next(s for s in result.trace if s.name == "classify")
    assert any("llm provider error" in d for d in classify_step.details)
    assert any("openai provider call failed" in d for d in classify_step.details)


def test_extract_failure_falls_back_with_a_recorded_note(fake_openai):
    FakeOpenAI.fail_kinds = {"extract"}
    backend = OpenAIBackend()
    extractions = extract_documents(_shipment(), backend)
    assert extractions[0].source == "provided"
    assert "LLM extraction failed" in extractions[0].note
    assert "openai provider call failed" in extractions[0].note


def test_diagnose_failure_falls_back_with_a_recorded_note(fake_openai):
    FakeOpenAI.fail_kinds = {"diagnose"}
    backend = OpenAIBackend()
    diagnosis = build_diagnosis(
        shipment=_shipment(),
        classification={
            "exception_type": "delay", "severity": "high", "rationale": "r", "signals": [],
        },
        delay_hours=36.0,
        mismatches=[],
        extractions=[],
        policies=[],
        backend=backend,
    )
    assert diagnosis.source == "template"
    assert "LLM diagnosis failed" in diagnosis.note


def test_options_failure_falls_back_with_a_recorded_note(fake_openai):
    from shipment_agent.model_backends import DraftContext
    from shipment_agent.options import build_recovery_options

    FakeOpenAI.fail_kinds = {"options"}
    notes: list[str] = []
    options = build_recovery_options(
        exception_type="delay",
        severity="high",
        delay_hours=36.0,
        backend=OpenAIBackend(),
        context=DraftContext(shipment_id="ERR-1"),
        notes=notes,
    )
    assert options  # template proposals still scored
    assert any("LLM option proposals failed" in n for n in notes)


# --------------------------------------------------------------------------
# Client *construction* failures are translated too (the NO_PROXY case)
# --------------------------------------------------------------------------

class InvalidURL(Exception):
    """Stands in for httpx.InvalidURL — raised while a client is being
    constructed when NO_PROXY carries entries the URL parser rejects
    (bracketed IPv6 like [::1]). Found by a live run: it escaped as a
    raw traceback because construction sits outside the call wrapping."""


class ExplodingOpenAI:
    def __init__(self, **kwargs):
        raise InvalidURL("Invalid port: ':1]'")


class ExplodingAnthropic:
    def __init__(self, **kwargs):
        raise InvalidURL("Invalid port: ':1]'")


@pytest.fixture
def exploding_openai(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=ExplodingOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return ExplodingOpenAI


@pytest.fixture
def exploding_anthropic(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "anthropic", SimpleNamespace(Anthropic=ExplodingAnthropic)
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    return ExplodingAnthropic


def test_openai_construction_failure_is_translated(exploding_openai):
    from shipment_agent.model_backends import OpenAIBackend

    with pytest.raises(ProviderError) as excinfo:
        OpenAIBackend()
    message = str(excinfo.value)
    assert "openai provider client construction failed" in message
    assert "NO_PROXY" in message  # the likely cause is named
    assert "Traceback" not in message


def test_anthropic_construction_failure_is_translated(exploding_anthropic):
    from shipment_agent.model_backends import AnthropicBackend

    with pytest.raises(ProviderError) as excinfo:
        AnthropicBackend()
    message = str(excinfo.value)
    assert "anthropic provider client construction failed" in message
    assert "NO_PROXY" in message


def test_ollama_construction_failure_is_translated(exploding_openai):
    with pytest.raises(ProviderError) as excinfo:
        OllamaBackend()
    assert "ollama provider client construction failed" in str(excinfo.value)


def test_embeddings_construction_failure_is_translated(exploding_openai):
    with pytest.raises(ProviderError) as excinfo:
        get_retriever("semantic")
    message = str(excinfo.value)
    assert "openai embeddings (RETRIEVER=semantic)" in message
    assert "construction failed" in message


def test_cli_reports_construction_failure_cleanly(exploding_openai, monkeypatch, capsys):
    monkeypatch.setenv("MODEL_BACKEND", "openai")
    monkeypatch.setattr("sys.argv", ["shipment-agent", "--index", "0"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: openai provider client construction failed")
    assert "Traceback" not in err


def test_api_returns_502_for_construction_failure(exploding_openai, monkeypatch):
    from shipment_agent.store import InMemoryStore

    monkeypatch.setenv("MODEL_BACKEND", "openai")
    service = ShipmentService(store=InMemoryStore())  # backend resolved per-call
    monkeypatch.setattr(api_module, "service", service)
    client = TestClient(api_module.app, raise_server_exceptions=False)
    response = client.post("/shipments/analyze", json=SHIPMENT)
    assert response.status_code == 502
    assert "construction failed" in response.json()["detail"]


# --------------------------------------------------------------------------
# Embeddings errors name the RETRIEVER value that was set
# --------------------------------------------------------------------------

def test_hybrid_embeddings_error_names_hybrid(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match=r"RETRIEVER=hybrid"):
        get_retriever("hybrid")


def test_semantic_embeddings_error_names_semantic(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match=r"RETRIEVER=semantic"):
        get_retriever("semantic")
