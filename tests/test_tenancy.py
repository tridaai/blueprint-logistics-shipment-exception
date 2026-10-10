"""Multi-tenant store partitioning.

One deployment serves many clients; the store is where their data
must never mix. Every record carries a ``tenant_id`` and the store's
identity is the pair ``(tenant_id, shipment_id)`` — two tenants may
both have a SYN-1001, every read is scoped to one partition, and a
cross-tenant read finds nothing (the API turns that into a 404).
These tests pin the isolation on both hermetic store backends
(in-memory + SQLite doubles), at the service layer (memory, queue,
scorecards, idempotency, decisions), and at the API (the
``X-Tenant-ID`` header). The Postgres store follows the same
contract in the gated integration tests at the bottom.
"""

from __future__ import annotations

import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import (
    ApprovalRecord,
    InMemoryStore,
    SQLiteStore,
)

DELAY_SHIPMENT = {
    "shipment_id": "TEN-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "carrier": "Synthetic Carrier",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


def _service(store=None) -> tuple[ShipmentService, MockModelBackend]:
    backend = MockModelBackend()
    service = ShipmentService(
        backend=backend,
        retriever=KeywordRetriever(),
        store=store or InMemoryStore(),
        checkpointer=False,
    )
    return service, backend


def _record(shipment_id: str, tenant_id: str, service_result=None) -> ApprovalRecord:
    """A minimal stored record for one tenant, analysed for real so
    the result is a genuine AgentResult."""
    service, _ = _service()
    payload = dict(DELAY_SHIPMENT, shipment_id=shipment_id)
    result = service_result or service.analyze(payload)
    return ApprovalRecord(result=result, tenant_id=tenant_id, shipment=payload)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryStore()
    return SQLiteStore(tmp_path / "approvals.db")


# ---------------------------------------------------------------------------
# Store-level isolation
# ---------------------------------------------------------------------------


def test_same_shipment_id_coexists_in_two_tenants(store):
    store.save(_record("TEN-1", "acme"))
    store.save(_record("TEN-1", "globex"))

    # (SQLite round-trips through JSON, so ownership is compared by
    # tenant, not object identity.)
    assert store.get("TEN-1", tenant_id="acme").tenant_id == "acme"
    assert store.get("TEN-1", tenant_id="globex").tenant_id == "globex"
    assert len(store.records(tenant_id="acme")) == 1
    assert len(store.records(tenant_id="globex")) == 1
    assert len(store.records()) == 2  # the operator's unscoped view


def test_cross_tenant_get_finds_nothing(store):
    store.save(_record("TEN-1", "acme"))
    assert store.get("TEN-1", tenant_id="globex") is None
    assert store.records(tenant_id="globex") == []


def test_history_and_feedback_reads_are_scoped(store):
    record = _record("TEN-1", "acme")
    record.approved = True
    record.approver = "ops-lead"
    record.approve_reason = "customer accepted the revised ETA"
    record.result.approval_status = "approved"
    store.save(record)

    assert len(store.prior_shipments(tenant_id="acme")) == 1
    assert store.prior_shipments(tenant_id="globex") == []
    assert len(store.decision_feedback(tenant_id="acme")) == 1
    assert store.decision_feedback(tenant_id="globex") == []


def test_idempotency_lookup_is_scoped(store):
    acme = _record("TEN-1", "acme")
    acme.idempotency_key = "intake-7"
    store.save(acme)

    found = store.get_by_idempotency("intake-7", "TEN-1", tenant_id="acme")
    assert found is not None and found.tenant_id == "acme"
    assert store.get_by_idempotency("intake-7", "TEN-1", tenant_id="globex") is None


def test_reanalysis_replaces_only_its_own_tenants_record(store):
    store.save(_record("TEN-1", "acme"))
    store.save(_record("TEN-1", "globex"))
    store.save(_record("TEN-1", "acme"))  # re-analyse acme's shipment

    assert len(store.records()) == 2
    assert store.get("TEN-1", tenant_id="globex").tenant_id == "globex"


def test_sqlite_pre_tenancy_database_is_rebuilt_with_rows_intact(tmp_path):
    """A database file written before tenancy (shipment_id the sole
    primary key, no tenant column) opens with its rows preserved in
    the default tenant and the composite key in place."""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE approvals (
            shipment_id TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            approver TEXT,
            approved INTEGER NOT NULL DEFAULT 0,
            rejected_by TEXT,
            reject_reason TEXT NOT NULL DEFAULT ''
        )
        """
    )
    service, _ = _service()
    result = service.analyze(dict(DELAY_SHIPMENT))
    conn.execute(
        "INSERT INTO approvals (shipment_id, result_json) VALUES (?, ?)",
        ("TEN-1", result.model_dump_json()),
    )
    conn.commit()
    conn.close()

    store = SQLiteStore(path)
    record = store.get("TEN-1", tenant_id="default")
    assert record is not None
    assert record.tenant_id == "default"
    # The composite key is live: another tenant can own the same id.
    store.save(_record("TEN-1", "acme", service_result=result))
    assert len(store.records()) == 2
    pk = {
        row[1]
        for row in sqlite3.connect(path).execute("PRAGMA table_info(approvals)")
        if row[5]
    }
    assert pk == {"tenant_id", "shipment_id"}


# ---------------------------------------------------------------------------
# Service-level isolation
# ---------------------------------------------------------------------------


def test_service_reads_and_decisions_stay_in_their_partition():
    service, _ = _service()
    service.analyze(dict(DELAY_SHIPMENT), tenant_id="acme")

    assert service.get("TEN-1", tenant_id="acme") is not None
    assert service.get("TEN-1", tenant_id="globex") is None
    with pytest.raises(KeyError):
        service.approve("TEN-1", approver="ops-lead", tenant_id="globex")
    decided = service.approve("TEN-1", approver="ops-lead", tenant_id="acme")
    assert decided.approval_status == "approved"


def test_queue_and_scorecards_partition_by_tenant():
    service, _ = _service()
    service.analyze(dict(DELAY_SHIPMENT), tenant_id="acme")
    service.analyze(
        dict(DELAY_SHIPMENT, shipment_id="TEN-2"), tenant_id="globex"
    )

    acme_queue = service.approval_queue(tenant_id="acme")
    assert [item["shipment_id"] for item in acme_queue] == ["TEN-1"]
    globex_queue = service.approval_queue(tenant_id="globex")
    assert [item["shipment_id"] for item in globex_queue] == ["TEN-2"]

    assert service.carrier_scorecard("Synthetic Carrier", tenant_id="acme")[
        "shipments"
    ] == 1
    assert service.carrier_scorecard("Synthetic Carrier", tenant_id="globex")[
        "shipments"
    ] == 1
    # A carrier unknown to a partition is unknown there, full stop.
    assert (
        service.carrier_scorecard("Other Carrier", tenant_id="acme") is None
    )


def test_memory_does_not_cross_tenants():
    """Tenant B's new analysis must not see tenant A's history: the
    priors the pipeline reads are the caller's own partition."""
    service, _ = _service()
    service.analyze(dict(DELAY_SHIPMENT), tenant_id="acme")
    store = service._get_store()
    assert store.prior_shipments(tenant_id="globex") == []
    # B analyses the same consignee/lane/carrier: its scorecard read
    # still counts only B's own record once saved.
    service.analyze(dict(DELAY_SHIPMENT, shipment_id="TEN-2"), tenant_id="globex")
    card_b = service.carrier_scorecard("Synthetic Carrier", tenant_id="globex")
    assert card_b["shipments"] == 1


def test_idempotency_scope_is_per_tenant():
    service, backend = _service()
    service.analyze(dict(DELAY_SHIPMENT), idempotency_key="k-1", tenant_id="acme")
    calls = backend.usage_totals()["calls"]

    # Same key, same shipment id, different tenant: a new submission,
    # not a replay — the pipeline runs.
    other = service.analyze(
        dict(DELAY_SHIPMENT), idempotency_key="k-1", tenant_id="globex"
    )
    assert other.idempotent_replay is False
    assert backend.usage_totals()["calls"] > calls

    # Each tenant's replay returns its own stored run — no new calls.
    calls_after_globex = backend.usage_totals()["calls"]
    replay = service.analyze(
        dict(DELAY_SHIPMENT), idempotency_key="k-1", tenant_id="acme"
    )
    assert replay.idempotent_replay is True
    assert backend.usage_totals()["calls"] == calls_after_globex


def test_tenant_id_env_is_the_process_default(monkeypatch):
    monkeypatch.setenv("TENANT_ID", "acme")
    service, _ = _service()
    service.analyze(dict(DELAY_SHIPMENT))  # no tenant argument anywhere

    store = service._get_store()
    assert len(store.records(tenant_id="acme")) == 1
    assert store.records(tenant_id="default") == []
    # And resolution follows the env for reads too.
    assert service.get("TEN-1") is not None
    assert service.get("TEN-1", tenant_id="default") is None


# ---------------------------------------------------------------------------
# API-level isolation (X-Tenant-ID)
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    service, _ = _service()
    monkeypatch.setattr(api_module, "service", service)
    return TestClient(api_module.app), service


def test_api_cross_tenant_read_is_a_404(client):
    http, _ = client
    created = http.post(
        "/shipments/analyze",
        json=DELAY_SHIPMENT,
        headers={"X-Tenant-ID": "acme"},
    )
    assert created.status_code == 200

    assert http.get("/shipments/TEN-1").status_code == 404
    assert (
        http.get("/shipments/TEN-1", headers={"X-Tenant-ID": "globex"}).status_code
        == 404
    )
    assert (
        http.get("/shipments/TEN-1", headers={"X-Tenant-ID": "acme"}).status_code
        == 200
    )


def test_api_queue_and_decisions_are_scoped(client):
    http, _ = client
    http.post(
        "/shipments/analyze", json=DELAY_SHIPMENT, headers={"X-Tenant-ID": "acme"}
    )
    queue = http.get("/queue", headers={"X-Tenant-ID": "globex"})
    assert queue.json()["count"] == 0
    queue = http.get("/queue", headers={"X-Tenant-ID": "acme"})
    assert queue.json()["count"] == 1

    denied = http.post(
        "/shipments/TEN-1/approve",
        json={"actor": "ops-lead"},
        headers={"X-Tenant-ID": "globex"},
    )
    assert denied.status_code == 404
    allowed = http.post(
        "/shipments/TEN-1/approve",
        json={"actor": "ops-lead"},
        headers={"X-Tenant-ID": "acme"},
    )
    assert allowed.status_code == 200


# ---------------------------------------------------------------------------
# Postgres (gated): the production store honours the same contract.
# ---------------------------------------------------------------------------

_DATABASE_URL = os.environ.get("DATABASE_URL")

pg = pytest.mark.skipif(
    not _DATABASE_URL,
    reason="DATABASE_URL not set — Postgres tenancy tests skipped",
)


@pg
def test_postgres_store_partitions_by_tenant(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _DATABASE_URL)
    monkeypatch.delenv("STATE_DB_PATH", raising=False)
    from shipment_agent.db import connect, run_migrations
    from shipment_agent.store import PostgresStore

    run_migrations(_DATABASE_URL)
    with connect(_DATABASE_URL) as conn:
        conn.execute("DELETE FROM approvals WHERE shipment_id = 'TEN-PG-1'")
        conn.commit()
    store = PostgresStore(_DATABASE_URL)
    service, _ = _service()
    result = service.analyze(dict(DELAY_SHIPMENT))
    try:
        store.save(
            ApprovalRecord(
                result=result.model_copy(update={"shipment_id": "TEN-PG-1"}),
                tenant_id="acme",
            )
        )
        store.save(
            ApprovalRecord(
                result=result.model_copy(update={"shipment_id": "TEN-PG-1"}),
                tenant_id="globex",
            )
        )
        assert store.get("TEN-PG-1", tenant_id="acme").tenant_id == "acme"
        assert store.get("TEN-PG-1", tenant_id="globex").tenant_id == "globex"
        assert len(store.records(tenant_id="acme")) >= 1
        assert all(
            r.tenant_id == "acme" for r in store.records(tenant_id="acme")
        )
    finally:
        with connect(_DATABASE_URL) as conn:
            conn.execute("DELETE FROM approvals WHERE shipment_id = 'TEN-PG-1'")
            conn.commit()
