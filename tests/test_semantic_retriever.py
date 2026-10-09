"""Semantic retriever tests — the embeddings API is faked (no network).

The fake embedding function maps text to a vector of keyword counts over
fixed dimensions, which makes cosine ranking deterministic and assertable.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.retriever import KeywordRetriever, SemanticRetriever, get_retriever
from shipment_agent.schemas import RetrievedPolicy

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_EMBEDDING_MODEL",
    "ANTHROPIC_API_KEY", "LLM_TIMEOUT_SECONDS", "RETRIEVER",
]

_DIMS = ("delay", "damage", "document", "appointment", "claim", "refund")


def _vec(text: str) -> list[float]:
    lowered = text.lower()
    return [float(lowered.count(word)) for word in _DIMS]


class _FakeEmbeddings:
    def create(self, model=None, input=None):
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=_vec(text))
                for i, text in enumerate(input)
            ]
        )


class FakeEmbeddingOpenAI:
    instances: list["FakeEmbeddingOpenAI"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.embeddings = _FakeEmbeddings()
        FakeEmbeddingOpenAI.instances.append(self)


POLICIES = [
    {"policy_id": "P-DELAY", "title": "Delay notification",
     "text": "delay delay customer delay update"},
    {"policy_id": "P-DAMAGE", "title": "Damage claims",
     "text": "damage claim packet photographs damage"},
    {"policy_id": "P-APPT", "title": "Appointments",
     "text": "appointment rescheduling dock appointment"},
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.model_backends.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.retriever.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def fake_embeddings(monkeypatch):
    FakeEmbeddingOpenAI.instances = []
    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(OpenAI=FakeEmbeddingOpenAI)
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return FakeEmbeddingOpenAI


def test_semantic_retriever_ranks_by_cosine(fake_embeddings):
    retriever = SemanticRetriever(policies=POLICIES)
    results = retriever.retrieve("delay delay customer update", top_k=3)
    assert all(isinstance(r, RetrievedPolicy) for r in results)
    assert results[0].policy_id == "P-DELAY"
    assert results[0].snippet == POLICIES[0]["text"]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].score > results[-1].score


def test_semantic_retriever_top_k(fake_embeddings):
    retriever = SemanticRetriever(policies=POLICIES)
    results = retriever.retrieve("damage claim", top_k=1)
    assert len(results) == 1
    assert results[0].policy_id == "P-DAMAGE"


def test_get_retriever_defaults_to_keyword():
    assert isinstance(get_retriever(), KeywordRetriever)
    assert isinstance(get_retriever("keyword"), KeywordRetriever)


def test_get_retriever_semantic_constructs(fake_embeddings):
    retriever = get_retriever("semantic")
    assert isinstance(retriever, SemanticRetriever)


def test_semantic_fails_loudly_when_anthropic_without_openai_key(monkeypatch):
    """Anthropic has no embeddings API — the error must say what IS needed."""
    monkeypatch.setenv("MODEL_BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    with pytest.raises(RuntimeError) as excinfo:
        get_retriever("semantic")
    message = str(excinfo.value)
    assert "OPENAI_API_KEY is not set" in message
    assert "embeddings" in message
    assert "anthropic" in message


def test_semantic_fails_loudly_when_mock_without_openai_key():
    with pytest.raises(RuntimeError) as excinfo:
        get_retriever("semantic")
    assert "OPENAI_API_KEY is not set" in str(excinfo.value)


def test_unknown_retriever_fails(monkeypatch):
    monkeypatch.setenv("RETRIEVER", "bogus")
    with pytest.raises(ValueError):
        get_retriever()
