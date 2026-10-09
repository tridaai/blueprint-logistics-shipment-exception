"""Chroma vector-store tests — chromadb is faked in ``sys.modules``.

The fake implements the documented PersistentClient surface the
retriever uses (get_or_create_collection / count / add / query with
cosine distance), so the Chroma path is exercised with fake embeddings
and no network. A separate test pins the fallback: no chromadb module
-> in-memory cosine serves, and ``vector_store`` reports "memory".
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

from shipment_agent.retriever import SemanticRetriever

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_EMBEDDING_MODEL",
    "ANTHROPIC_API_KEY", "LLM_TIMEOUT_SECONDS", "RETRIEVER", "CHROMA_DIR",
]

_DIMS = ("delay", "damage", "document", "appointment", "claim", "refund")


def _vec(text: str) -> list[float]:
    lowered = text.lower()
    return [float(lowered.count(word)) for word in _DIMS]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 - (dot / norm if norm else 0.0)


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


class FakeCollection:
    def __init__(self) -> None:
        self.rows: dict[str, list[float]] = {}
        self.add_calls = 0

    def count(self) -> int:
        return len(self.rows)

    def delete(self, ids=None):
        for pid in ids or []:
            self.rows.pop(pid, None)

    def add(self, ids=None, embeddings=None, documents=None, metadatas=None):
        self.add_calls += 1
        for pid, vector in zip(ids, embeddings):
            self.rows[pid] = list(vector)

    def query(self, query_embeddings=None, n_results=3):
        query = query_embeddings[0]
        ranked = sorted(
            self.rows.items(), key=lambda kv: _cosine_distance(query, kv[1])
        )[:n_results]
        return {
            "ids": [[pid for pid, _ in ranked]],
            "distances": [[_cosine_distance(query, v) for _, v in ranked]],
        }


class FakeChromaClient:
    instances: list["FakeChromaClient"] = []

    def __init__(self, path=None) -> None:
        self.path = path
        self.collection = FakeCollection()
        FakeChromaClient.instances.append(self)

    def get_or_create_collection(self, name, metadata=None):
        assert name.startswith("policies-")  # corpus-fingerprinted name
        assert metadata == {"hnsw:space": "cosine"}
        return self.collection


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
    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(OpenAI=FakeEmbeddingOpenAI)
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


@pytest.fixture
def fake_chroma(monkeypatch, tmp_path):
    FakeChromaClient.instances = []
    monkeypatch.setitem(
        sys.modules, "chromadb", SimpleNamespace(PersistentClient=FakeChromaClient)
    )
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    return FakeChromaClient


def test_chroma_path_indexes_once_and_ranks(fake_chroma):
    retriever = SemanticRetriever(policies=POLICIES)
    results = retriever.retrieve("delay delay customer update", top_k=3)
    assert retriever.vector_store == "chroma"
    assert results[0].policy_id == "P-DELAY"
    assert results[0].retrieval == "semantic"
    # Cosine similarity recovered from Chroma's distance (1 - distance).
    assert 0.9 < results[0].score <= 1.0
    client = fake_chroma.instances[-1]
    assert client.path.endswith("chroma")  # CHROMA_DIR honoured
    assert client.collection.add_calls == 1  # corpus indexed on first use
    # Second retrieve reuses the store — no re-indexing.
    retriever.retrieve("damage claim packet", top_k=1)
    assert client.collection.add_calls == 1


def test_memory_fallback_when_chromadb_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", None)  # import raises ImportError
    retriever = SemanticRetriever(policies=POLICIES)
    results = retriever.retrieve("delay delay customer update", top_k=3)
    assert retriever.vector_store == "memory"
    assert results[0].policy_id == "P-DELAY"


class FakeChromaHttpClient(FakeChromaClient):
    """Records the host/port the retriever dialed (CHROMA_HOST mode)."""

    dialed: list[tuple[str, int]] = []

    def __init__(self, host=None, port=None) -> None:
        super().__init__(path=None)
        FakeChromaHttpClient.dialed.append((host, port))


def test_chroma_server_mode_via_env(monkeypatch):
    """CHROMA_HOST set (the compose local stack): HttpClient, not files."""
    FakeChromaHttpClient.dialed = []
    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        SimpleNamespace(
            PersistentClient=FakeChromaClient, HttpClient=FakeChromaHttpClient
        ),
    )
    monkeypatch.setenv("CHROMA_HOST", "chroma")
    monkeypatch.setenv("CHROMA_PORT", "8000")
    monkeypatch.delenv("CHROMA_DIR", raising=False)
    retriever = SemanticRetriever(policies=POLICIES)
    results = retriever.retrieve("delay delay customer update", top_k=1)
    assert retriever.vector_store == "chroma"
    assert results[0].policy_id == "P-DELAY"
    assert FakeChromaHttpClient.dialed == [("chroma", 8000)]
    assert FakeChromaClient.instances[-1].path is None  # no local dir used
