"""Tenant corpus management: a tenant's operators add, replace, and
remove their own policy documents at runtime.

Bundled corpora are code; onboarding a real tenant cannot wait for a
deploy every time its SOPs change. Documents written through the
surface are persisted through the store port (the system of record),
archived through the object-store port when one is configured, and
merged over the bundled corpus per run — so retrieval picks them up
with no restart, scoped to their tenant exactly like the bundled
SOPs. These tests pin the store contract on both hermetic backends,
the merge/provenance rules, the write path's refusals, retrieval
without restart (and its absence for other tenants), and the API.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.object_store import InMemoryObjectStore
from shipment_agent.retriever import KeywordRetriever, merge_corpus
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore, SQLiteStore

FORKLIFT_SHIPMENT = {
    "shipment_id": "MGT-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "carrier": "Synthetic Carrier",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": (
        "Forklift punctured the shrink wrap on a pallet of bottled "
        "water at the cross-dock"
    ),
    "documents": [],
}

FORKLIFT_DOC = {
    "policy_id": "SOP-ACME-FORKLIFT",
    "title": "Forklift puncture quarantine (Acme)",
    "text": (
        "When a forklift punctures shrink wrap on a pallet, quarantine "
        "the pallet, inspect every bottled unit for leaks, and photograph "
        "the puncture before the load moves."
    ),
}


def _service(store=None, object_store=False) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store if store is not None else InMemoryStore(),
        checkpointer=False,
        object_store=object_store,
    )


# ---------------------------------------------------------------------------
# The merge rule
# ---------------------------------------------------------------------------


def test_merge_corpus_replaces_in_place_and_appends():
    base = [
        {"policy_id": "A", "title": "a", "text": "one"},
        {"policy_id": "B", "title": "b", "text": "two"},
    ]
    merged = merge_corpus(
        base,
        [
            {"policy_id": "B", "title": "b2", "text": "revised"},
            {"policy_id": "C", "title": "c", "text": "three"},
        ],
    )
    assert [p["policy_id"] for p in merged] == ["A", "B", "C"]
    assert merged[1]["text"] == "revised"
    # The base corpus is not mutated by the merge.
    assert base[1]["text"] == "two"


def test_retriever_with_extra_policies_merges_and_scopes():
    base = KeywordRetriever()
    view = base.for_tenant("acme").with_extra_policies(
        [{**FORKLIFT_DOC, "tenant_id": "acme"}]
    )
    hits = view.retrieve("forklift punctured shrink wrap pallet", top_k=5)
    assert "SOP-ACME-FORKLIFT" in [p.policy_id for p in hits]
    # Scoping still applies to stored documents: globex's view of the
    # same merged corpus does not contain acme's document.
    globex = base.for_tenant("globex").with_extra_policies(
        [{**FORKLIFT_DOC, "tenant_id": "acme"}]
    )
    hits = globex.retrieve("forklift punctured shrink wrap pallet", top_k=5)
    assert "SOP-ACME-FORKLIFT" not in [p.policy_id for p in hits]
    # An explicitly injected corpus is the injector's own — unchanged.
    explicit = KeywordRetriever([{"policy_id": "X", "title": "x", "text": "y"}])
    assert explicit.with_extra_policies([FORKLIFT_DOC]) is explicit


# ---------------------------------------------------------------------------
# The store contract (both hermetic backends)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def doc_store(request, tmp_path):
    if request.param == "memory":
        return InMemoryStore()
    return SQLiteStore(tmp_path / "state.db")


def test_store_round_trips_tenant_documents(doc_store):
    doc = {**FORKLIFT_DOC, "tenant_id": "acme", "updated_at": "2026-10-10T00:00:00"}
    doc_store.save_tenant_policy("acme", doc)
    assert doc_store.tenant_policy("acme", "SOP-ACME-FORKLIFT") == doc
    assert doc_store.tenant_policies("acme") == [doc]
    # Partitioned like everything else: globex sees nothing.
    assert doc_store.tenant_policies("globex") == []
    assert doc_store.tenant_policy("globex", "SOP-ACME-FORKLIFT") is None
    assert doc_store.delete_tenant_policy("acme", "SOP-ACME-FORKLIFT") is True
    assert doc_store.delete_tenant_policy("acme", "SOP-ACME-FORKLIFT") is False
    assert doc_store.tenant_policies("acme") == []


# ---------------------------------------------------------------------------
# The service: provenance, refusals, lifecycle
# ---------------------------------------------------------------------------


def test_corpus_listing_carries_provenance():
    service = _service()
    listing = service.corpus_policies("acme")
    by_id = {p["policy_id"]: p for p in listing}
    assert by_id["POL-DELAY-01"]["source"] == "shared"
    assert by_id["SOP-ACME-01"]["source"] == "bundled"
    service.upsert_tenant_policy("acme", **FORKLIFT_DOC)
    by_id = {p["policy_id"]: p for p in service.corpus_policies("acme")}
    assert by_id["SOP-ACME-FORKLIFT"]["source"] == "tenant"
    assert by_id["SOP-ACME-FORKLIFT"]["tenant_id"] == "acme"
    # Another tenant's listing never contains it.
    globex_ids = [p["policy_id"] for p in service.corpus_policies("globex")]
    assert "SOP-ACME-FORKLIFT" not in globex_ids


def test_upsert_refuses_empty_fields_and_shared_ids():
    service = _service()
    with pytest.raises(ValueError):
        service.upsert_tenant_policy("acme", "", "Title", "Text")
    with pytest.raises(ValueError):
        service.upsert_tenant_policy("acme", "SOP-X", "  ", "Text")
    with pytest.raises(ValueError):
        service.upsert_tenant_policy("acme", "SOP-X", "Title", "")
    with pytest.raises(ValueError, match="shared"):
        service.upsert_tenant_policy("acme", "POL-DELAY-01", "Mine now", "Text")


def test_stored_override_replaces_bundled_until_removed():
    service = _service()
    service.upsert_tenant_policy(
        "acme", "SOP-ACME-01", "Revised reefer response", "Revised text."
    )
    entry = next(
        p for p in service.corpus_policies("acme") if p["policy_id"] == "SOP-ACME-01"
    )
    assert entry["source"] == "tenant" and entry["title"] == "Revised reefer response"
    # Removing the override resurfaces the bundled original.
    assert service.remove_tenant_policy("acme", "SOP-ACME-01") is True
    entry = next(
        p for p in service.corpus_policies("acme") if p["policy_id"] == "SOP-ACME-01"
    )
    assert entry["source"] == "bundled"
    assert "temperature excursion" in entry["title"]
    # A bundled document with no stored override cannot be removed.
    assert service.remove_tenant_policy("acme", "SOP-ACME-01") is False


def test_upsert_archives_through_the_object_store():
    objects = InMemoryObjectStore()
    service = _service(object_store=objects)
    stored = service.upsert_tenant_policy("acme", **FORKLIFT_DOC)
    key = "tenants/acme/policies/SOP-ACME-FORKLIFT.json"
    assert stored["object_key"] == key
    assert objects.exists(key)
    # Removal deletes the archive copy too.
    assert service.remove_tenant_policy("acme", "SOP-ACME-FORKLIFT") is True
    assert not objects.exists(key)


# ---------------------------------------------------------------------------
# Retrieval picks documents up without a restart
# ---------------------------------------------------------------------------


def test_a_stored_document_is_retrievable_by_the_next_run():
    service = _service()
    before = service.analyze(FORKLIFT_SHIPMENT, tenant_id="acme")
    assert "SOP-ACME-FORKLIFT" not in [p.policy_id for p in before.policies]
    service.upsert_tenant_policy("acme", **FORKLIFT_DOC)
    after = service.analyze(FORKLIFT_SHIPMENT, tenant_id="acme")
    assert "SOP-ACME-FORKLIFT" in [p.policy_id for p in after.policies]
    # Only for its tenant.
    other = service.analyze(FORKLIFT_SHIPMENT, tenant_id="globex")
    assert "SOP-ACME-FORKLIFT" not in [p.policy_id for p in other.policies]
    # And removal takes it back out of the corpus.
    service.remove_tenant_policy("acme", "SOP-ACME-FORKLIFT")
    gone = service.analyze(FORKLIFT_SHIPMENT, tenant_id="acme")
    assert "SOP-ACME-FORKLIFT" not in [p.policy_id for p in gone.policies]


# ---------------------------------------------------------------------------
# The API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api_module, "service", _service())
    return TestClient(api_module.app)


def test_api_adds_lists_and_removes_a_document(client):
    headers = {"X-Tenant-ID": "acme"}
    created = client.post("/policies", json=FORKLIFT_DOC, headers=headers)
    assert created.status_code == 201
    assert created.json()["source"] == "tenant"
    listing = client.get("/policies", headers=headers).json()
    entry = next(p for p in listing if p["policy_id"] == "SOP-ACME-FORKLIFT")
    assert entry["source"] == "tenant"
    removed = client.delete("/policies/SOP-ACME-FORKLIFT", headers=headers)
    assert removed.status_code == 200
    listing = client.get("/policies", headers=headers).json()
    assert "SOP-ACME-FORKLIFT" not in [p["policy_id"] for p in listing]


def test_api_write_is_scoped_to_the_callers_tenant(client):
    client.post("/policies", json=FORKLIFT_DOC, headers={"X-Tenant-ID": "acme"})
    globex = client.get("/policies", headers={"X-Tenant-ID": "globex"}).json()
    assert "SOP-ACME-FORKLIFT" not in [p["policy_id"] for p in globex]
    # Globex cannot delete what it cannot see.
    response = client.delete(
        "/policies/SOP-ACME-FORKLIFT", headers={"X-Tenant-ID": "globex"}
    )
    assert response.status_code == 404


def test_api_refusals(client):
    headers = {"X-Tenant-ID": "acme"}
    # Bundled documents are managed in code, not through the API.
    assert client.delete("/policies/SOP-ACME-01", headers=headers).status_code == 404
    # A shared document cannot be redefined by a tenant.
    response = client.post(
        "/policies",
        json={"policy_id": "POL-DELAY-01", "title": "Mine", "text": "Text"},
        headers=headers,
    )
    assert response.status_code == 422
    # Empty fields are refused, not stored.
    response = client.post(
        "/policies",
        json={"policy_id": "SOP-X", "title": "", "text": "Text"},
        headers=headers,
    )
    assert response.status_code == 422
