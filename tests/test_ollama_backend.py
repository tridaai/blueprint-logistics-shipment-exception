"""Ollama backend tests — the OpenAI SDK is faked; no network, no server.

Ollama is a preset over the OpenAI-compatible path: a local base URL and
a placeholder key. These tests assert the preset is wired correctly for
both chat (drafting + cross-check) and embeddings (semantic retrieval),
with NO cloud API key anywhere in the environment.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import OllamaBackend, get_backend
from shipment_agent.retriever import SemanticRetriever
from shipment_agent.schemas import ShipmentInput

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "OPENAI_EMBEDDING_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
    "ANTHROPIC_BASE_URL", "LLM_TIMEOUT_SECONDS", "RETRIEVER",
    "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_EMBEDDING_MODEL",
]

DELAY_SHIPMENT = {
    "shipment_id": "OLL-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

DRAFT_TEXT = (
    "Subject: Update on shipment OLL-1: delay\n\n"
    "Dear Synthetic Customer,\n\n"
    "Your shipment OLL-1 is delayed. [POL-DELAY-01] applies. "
    "The next update will arrive within one business day."
)
CLASSIFY_TEXT = (
    '{"exception_type": "delay", "severity": "high", "confidence": 0.9,'
    ' "rationale": "Computed delay dominates."}'
)

_DIMS = ("delay", "damage", "document", "appointment", "claim", "refund")


def _vec(text: str) -> list[float]:
    lowered = text.lower()
    return [float(lowered.count(word)) for word in _DIMS]


class _FakeCompletions:
    def __init__(self, client: "FakeOllamaOpenAI") -> None:
        self._client = client

    def create(self, model=None, max_tokens=None, messages=None):
        system = " ".join(messages[0]["content"].split())
        self._client.calls.append({"model": model})
        text = CLASSIFY_TEXT if "classifying a shipment exception" in system else DRAFT_TEXT
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class _FakeEmbeddings:
    def __init__(self, client: "FakeOllamaOpenAI") -> None:
        self._client = client

    def create(self, model=None, input=None):
        self._client.embedding_calls.append({"model": model})
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=_vec(t)) for i, t in enumerate(input)]
        )


class FakeOllamaOpenAI:
    instances: list["FakeOllamaOpenAI"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[dict] = []
        self.embedding_calls: list[dict] = []
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))
        self.embeddings = _FakeEmbeddings(self)
        FakeOllamaOpenAI.instances.append(self)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.model_backends.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.retriever.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def fake_ollama(monkeypatch):
    FakeOllamaOpenAI.instances = []
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOllamaOpenAI))
    return FakeOllamaOpenAI


def test_ollama_backend_needs_no_api_key(fake_ollama):
    backend = get_backend("ollama")
    assert isinstance(backend, OllamaBackend)
    assert backend.name == "ollama"
    kwargs = fake_ollama.instances[-1].kwargs
    assert kwargs["base_url"] == "http://localhost:11434/v1"
    assert kwargs["api_key"] == "ollama"  # placeholder, ignored by Ollama


def test_ollama_base_url_and_model_overrides(fake_ollama, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:14b")
    backend = get_backend("ollama")
    assert fake_ollama.instances[-1].kwargs["base_url"] == "http://gpu-box:11434/v1"
    result = run_shipment(ShipmentInput.model_validate(DELAY_SHIPMENT), backend=backend)
    models = {call["model"] for call in fake_ollama.instances[-1].calls}
    assert models == {"qwen2.5:14b"}
    assert result.validation.passed
    assert result.cross_check is not None
    assert result.cross_check.llm_backend == "ollama"
    assert result.cross_check.resolution == "agree"


def test_ollama_selected_via_env(fake_ollama, monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "ollama")
    assert isinstance(get_backend(), OllamaBackend)


def test_ollama_embeddings_for_semantic_retrieval(fake_ollama, monkeypatch):
    """MODEL_BACKEND=ollama: embeddings come from Ollama — no OpenAI key."""
    monkeypatch.setenv("MODEL_BACKEND", "ollama")
    policies = [
        {"policy_id": "P-DELAY", "title": "Delay notification",
         "text": "delay delay customer delay update"},
        {"policy_id": "P-DAMAGE", "title": "Damage claims",
         "text": "damage claim packet photographs damage"},
    ]
    retriever = SemanticRetriever(policies=policies)
    results = retriever.retrieve("delay delay customer update", top_k=1)
    assert results[0].policy_id == "P-DELAY"
    client = fake_ollama.instances[-1]
    assert client.kwargs["base_url"] == "http://localhost:11434/v1"
    assert {c["model"] for c in client.embedding_calls} == {"nomic-embed-text"}


def test_ollama_embedding_model_override(fake_ollama, monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
    retriever = SemanticRetriever(policies=[
        {"policy_id": "P1", "title": "Delay", "text": "delay delay"},
    ])
    retriever.retrieve("delay", top_k=1)
    client = fake_ollama.instances[-1]
    assert {c["model"] for c in client.embedding_calls} == {"mxbai-embed-large"}
