"""Per-tenant policy corpora: a tenant retrieves its own SOPs, and
only its own.

The store was partitioned in round 5; retrieval was the last shared
surface. The corpus is now tagged — shared documents carry no
tenant, each tenant's SOPs carry theirs — and every retriever
scopes by tenant: a run retrieves the shared corpus plus its own
tenant's documents, while another tenant's documents are *absent
from the corpus*, not filtered from the results. These tests pin
the scoping at the corpus helper, at each retriever, end to end
through the service (a tenant SOP retrieved only for that tenant),
and at the /policies listing. The pgvector tenant filter is pinned
hermetically over a fake connection; the live round-trip stays in
the gated Postgres integration tests.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.demo import TENANT_DEMO_SHIPMENT
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.policies_data import POLICIES, TENANT_POLICIES
from shipment_agent.retriever import (
    HybridRetriever,
    KeywordRetriever,
    SemanticRetriever,
    corpus_for_tenant,
    full_corpus,
)
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

REEFER_QUERY = (
    "damage reefer unit alarm cross-dock temperature excursion above "
    "the setpoint, reefer telemetry logged the excursion, trailer released"
)
SEAL_QUERY = "security seal mismatch high-value load trailer sealed handover"

DELAY_SHIPMENT = {
    "shipment_id": "CORP-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "carrier": "Synthetic Carrier",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


def _ids(results) -> list[str]:
    return [policy.policy_id for policy in results]


# ---------------------------------------------------------------------------
# The corpus helper
# ---------------------------------------------------------------------------


def test_corpus_for_tenant_slices_the_tagged_corpus():
    corpus = full_corpus()
    acme = corpus_for_tenant(corpus, "acme")
    assert "SOP-ACME-01" in [p["policy_id"] for p in acme]
    assert "SOP-GLOBEX-01" not in [p["policy_id"] for p in acme]
    assert all(p["policy_id"].startswith("POL-") or p["tenant_id"] == "acme" for p in acme)
    # No tenant claimed: shared documents only, never a tenant's.
    anonymous = corpus_for_tenant(corpus, None)
    assert [p["policy_id"] for p in anonymous] == [p["policy_id"] for p in POLICIES]
    # A tenant with no documents of its own gets exactly the shared corpus.
    assert corpus_for_tenant(corpus, "initech") == anonymous


def test_tenant_policies_sample_file_mirrors_the_module():
    mirror = Path(__file__).resolve().parents[1] / "data" / "sample" / "tenant_policies.json"
    assert json.loads(mirror.read_text(encoding="utf-8")) == TENANT_POLICIES


# ---------------------------------------------------------------------------
# Retriever scoping
# ---------------------------------------------------------------------------


def test_keyword_retriever_scopes_by_tenant():
    base = KeywordRetriever()
    # Unscoped (the CLI/demo default): the shared corpus only.
    assert "SOP-ACME-01" not in _ids(base.retrieve(REEFER_QUERY))
    acme = base.for_tenant("acme")
    assert acme.tenant_id == "acme"
    assert "SOP-ACME-01" in _ids(acme.retrieve(REEFER_QUERY))
    # Globex's corpus does not contain acme's SOP at all.
    globex = base.for_tenant("globex")
    assert "SOP-ACME-01" not in _ids(globex.retrieve(REEFER_QUERY))
    assert "SOP-GLOBEX-01" in _ids(globex.retrieve(SEAL_QUERY))
    # And acme cannot reach globex's document either.
    assert "SOP-GLOBEX-01" not in _ids(acme.retrieve(SEAL_QUERY))


def test_shared_documents_rank_for_every_tenant():
    delay_query = "delay estimated delivery hours after scheduled proactive notification"
    for tenant in (None, "acme", "globex", "initech"):
        view = KeywordRetriever().for_tenant(tenant)
        assert "POL-DELAY-01" in _ids(view.retrieve(delay_query, top_k=3))


def test_explicit_corpus_is_the_injectors_own():
    """Evals and tests inject corpora; for_tenant must not rescope
    what was explicitly handed over."""
    retriever = KeywordRetriever(POLICIES)
    assert retriever.for_tenant("acme") is retriever
    assert "SOP-ACME-01" not in _ids(retriever.retrieve(REEFER_QUERY, top_k=10))


def test_hybrid_retriever_scopes_both_halves(monkeypatch):
    """The hybrid view rescopes keyword AND semantic; embeddings are
    faked the way test_hybrid_retriever fakes them (no network)."""
    import sys

    class _FakeEmbeddings:
        def create(self, model=None, input=None):
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=i, embedding=[float(len(text)), 1.0])
                    for i, text in enumerate(input)
                ]
            )

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.embeddings = _FakeEmbeddings()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setitem(sys.modules, "chromadb", None)  # in-memory fallback
    monkeypatch.setattr("shipment_agent.retriever.load_dotenv", lambda *a, **k: None)

    hybrid = HybridRetriever()
    view = hybrid.for_tenant("acme")
    assert view is not hybrid
    assert view.tenant_id == "acme"
    assert "SOP-ACME-01" in _ids(view._keyword.retrieve(REEFER_QUERY))
    assert view._semantic.tenant_id == "acme"
    assert "SOP-ACME-01" in _ids(view._semantic.retrieve(REEFER_QUERY, top_k=10))
    assert "SOP-ACME-01" not in _ids(
        hybrid.for_tenant("globex")._semantic.retrieve(REEFER_QUERY, top_k=10)
    )
    assert hybrid.for_tenant(None) is hybrid


# ---------------------------------------------------------------------------
# End to end through the service
# ---------------------------------------------------------------------------


def _service(retriever=None) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=retriever if retriever is not None else KeywordRetriever(),
        store=InMemoryStore(),
        checkpointer=False,
    )


def test_tenant_sop_is_retrieved_only_for_that_tenant():
    service = _service()
    acme = service.analyze(TENANT_DEMO_SHIPMENT, tenant_id="acme")
    globex = service.analyze(TENANT_DEMO_SHIPMENT, tenant_id="globex")
    default = service.analyze(TENANT_DEMO_SHIPMENT)
    assert "SOP-ACME-01" in [p.policy_id for p in acme.policies]
    assert "SOP-ACME-01" not in [p.policy_id for p in globex.policies]
    assert "SOP-ACME-01" not in [p.policy_id for p in default.policies]
    assert "SOP-GLOBEX-01" not in [p.policy_id for p in acme.policies]


def test_shared_corpus_still_serves_tenant_runs():
    service = _service()
    result = service.analyze(DELAY_SHIPMENT, tenant_id="acme")
    assert "POL-DELAY-01" in [p.policy_id for p in result.policies]


def test_a_retriever_without_for_tenant_is_used_as_provided():
    """The optional-capability contract: a custom retriever that
    does not offer tenant views is the injector's own seam — the
    service must not fail, and must not pretend to scope it."""

    class SharedOnlyRetriever:
        name = "shared-only"

        def retrieve(self, query, top_k=3):
            return KeywordRetriever(POLICIES).retrieve(query, top_k=top_k)

    service = _service(retriever=SharedOnlyRetriever())
    result = service.analyze(DELAY_SHIPMENT, tenant_id="acme")
    assert "POL-DELAY-01" in [p.policy_id for p in result.policies]


# ---------------------------------------------------------------------------
# The /policies listing is the caller's corpus
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api_module, "service", _service())
    return TestClient(api_module.app)


def test_policies_endpoint_lists_the_callers_corpus(client):
    shared = client.get("/policies")
    assert [p["policy_id"] for p in shared.json()] == [
        p["policy_id"] for p in POLICIES
    ]
    acme = client.get("/policies", headers={"X-Tenant-ID": "acme"})
    acme_ids = [p["policy_id"] for p in acme.json()]
    assert "SOP-ACME-01" in acme_ids and "SOP-GLOBEX-01" not in acme_ids
    sop = next(p for p in acme.json() if p["policy_id"] == "SOP-ACME-01")
    assert sop["tenant_id"] == "acme"  # the tag is on the document, in the open


# ---------------------------------------------------------------------------
# The pgvector tenant filter (hermetic: a fake connection)
# ---------------------------------------------------------------------------


class _FakeRows:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Just enough connection for _pg_ensure_corpus + the query:
    the corpus reads as fully embedded (matching content hashes),
    and the search returns one tenant row — which the retriever
    must still refuse to surface for another tenant's view."""

    def __init__(self, policies):
        self.queries: list[tuple[str, tuple]] = []
        self._hashes = {
            p["policy_id"]: hashlib.sha256(
                f"{p['policy_id']}|{p['title']}|{p['text']}".encode("utf-8")
            ).hexdigest()
            for p in policies
        }

    def execute(self, sql, params=()):
        self.queries.append((sql, params))
        if "SELECT policy_id, content_hash" in sql:
            return _FakeRows(sorted(self._hashes.items()))
        if "SELECT policy_id, 1 -" in sql:
            return _FakeRows([("SOP-ACME-01", 0.91)])
        raise AssertionError(f"unexpected SQL: {sql}")

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _StubEmbeddings:
    def create(self, model=None, input=None):
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=[0.1, 0.2])
                for i, _ in enumerate(input)
            ]
        )


class _StubClient:
    embeddings = _StubEmbeddings()


class _StubSemantic(SemanticRetriever):
    def _build_embeddings_client(self):
        return _StubClient(), "stub-model", "stub", "local"


def test_pgvector_query_filters_on_the_tenant_axis(monkeypatch):
    retriever = _StubSemantic()
    retriever._pgvector_checked = True
    retriever._pgvector_ready = True
    conn = _FakeConn(retriever._corpus)
    monkeypatch.setattr(
        "shipment_agent.db.connect", lambda *a, **k: conn
    )

    acme = retriever.for_tenant("acme")
    hits = acme._retrieve_pgvector("reefer excursion", top_k=3)
    assert _ids(hits) == ["SOP-ACME-01"]
    search_sql, search_params = next(
        (sql, params) for sql, params in conn.queries if "SELECT policy_id, 1 -" in sql
    )
    assert "tenant_id IS NOT DISTINCT FROM" in search_sql
    assert "acme" in search_params

    # The same stored row is invisible to globex's view: the SQL
    # filter scopes the candidates AND the corpus check drops any
    # row this view's corpus does not contain.
    globex = retriever.for_tenant("globex")
    assert globex._retrieve_pgvector("reefer excursion", top_k=3) == []
