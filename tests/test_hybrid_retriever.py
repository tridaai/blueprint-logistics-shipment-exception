"""Hybrid retrieval tests — keyword + semantic merge, then the rerank
step (reciprocal-rank score fusion). Embeddings are faked; no network.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import HybridRetriever, get_retriever, rerank_fused
from shipment_agent.schemas import RetrievedPolicy, ShipmentInput

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
    {"policy_id": "P-CLAIM", "title": "Claim filing",
     "text": "claim claim form deadline claim"},
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
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeEmbeddingOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return FakeEmbeddingOpenAI


def _policy(pid: str, score: float) -> RetrievedPolicy:
    return RetrievedPolicy(policy_id=pid, title=pid, snippet="text", score=score)


# --------------------------------------------------------------------------
# The rerank step itself
# --------------------------------------------------------------------------

def test_rerank_fused_dedupes_and_rewards_agreement():
    keyword = [_policy("A", 0.9), _policy("B", 0.5), _policy("C", 0.4)]
    semantic = [_policy("B", 0.8), _policy("A", 0.7), _policy("D", 0.6)]
    merged = rerank_fused(keyword, semantic, top_k=3)
    ids = [p.policy_id for p in merged]
    # A and B appear in both lists -> they take the top fused ranks.
    assert set(ids[:2]) == {"A", "B"}
    assert ids[2] in {"C", "D"}
    by_id = {p.policy_id: p for p in merged}
    assert by_id["A"].retrieval == "keyword+semantic"
    assert by_id["B"].retrieval == "keyword+semantic"
    # The returned score is the fused RRF score, not either raw score.
    assert by_id["A"].score == round(1 / 61 + 1 / 62, 4)


def test_rerank_fused_single_source_labels():
    merged = rerank_fused([_policy("A", 0.9)], [_policy("B", 0.9)], top_k=2)
    by_id = {p.policy_id: p for p in merged}
    assert by_id["A"].retrieval == "keyword"
    assert by_id["B"].retrieval == "semantic"
    scores = [p.score for p in merged]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------
# Hybrid retriever end to end (fake embeddings)
# --------------------------------------------------------------------------

def test_hybrid_merges_pools_and_reranks(fake_embeddings):
    retriever = HybridRetriever(policies=POLICIES)
    results = retriever.retrieve("delay delay customer update", top_k=3)
    assert results[0].policy_id == "P-DELAY"
    assert results[0].retrieval == "keyword+semantic"
    assert retriever.last_stats["keyword_pool"] >= 1
    assert retriever.last_stats["semantic_pool"] >= 1
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_get_retriever_hybrid_constructs(fake_embeddings):
    retriever = get_retriever("hybrid")
    assert isinstance(retriever, HybridRetriever)
    assert retriever.name == "hybrid"


def test_hybrid_fails_loudly_without_embeddings_key():
    with pytest.raises(RuntimeError) as excinfo:
        get_retriever("hybrid")
    assert "OPENAI_API_KEY is not set" in str(excinfo.value)


def test_graph_trace_shows_merge_and_rerank(fake_embeddings):
    retriever = HybridRetriever(policies=POLICIES)
    shipment = ShipmentInput.model_validate({
        "shipment_id": "HYB-1",
        "origin": "Memphis, TN",
        "destination": "Charlotte, NC",
        "scheduled_delivery": "2026-10-10T09:00:00",
        "estimated_delivery": "2026-10-11T21:00:00",
        "latest_event": "Delayed at regional hub due to weather hold",
        "documents": [],
    })
    result = run_shipment(shipment, backend=MockModelBackend(), retriever=retriever)
    assert result.policies
    step = next(s for s in result.trace if s.name == "retrieve")
    assert "mode=hybrid" in step.details[0]
    assert any("merge:" in d for d in step.details)
    assert any("rerank: reciprocal-rank" in d for d in step.details)
