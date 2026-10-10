"""Per-tenant worker watches + GET /workers: the worker picture,
split by the tenant whose work it is.

Worker staleness was deployment-wide: the retry worker could be
alive for the fleet while one tenant's retries rotted unswept.
The status rows now stamp ``by_tenant_last_at`` (the last sweep
that recorded an outcome for each tenant), and the staleness
judgement splits along it — same per-worker threshold, per-tenant
evidence. ``GET /workers`` serves the whole picture as JSON
(status rows, global verdicts, open episodes, the per-tenant
split), /metrics carries per-tenant families, and the escalation
digest's stale-workers section reads the same view. These tests
pin the stamp, the split's arithmetic, and all three surfaces.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.metrics import worker_tenant_staleness
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
OLD = "2026-10-08T12:00:00+00:00"  # 48h before NOW
FRESH = "2026-10-10T11:30:00+00:00"  # 30m before NOW


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("WORKER_STALE_SECONDS", "WORKER_STALE_SECONDS_DISPATCH_RETRIES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.service.load_dotenv", lambda *a, **k: None)


def _service(store=None) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store if store is not None else InMemoryStore(),
        checkpointer=False,
    )


def _row(**overrides):
    row = {
        "worker": "dispatch_retries",
        "sweeps": 3,
        "last_sweep_at": FRESH,
        "totals": {"attempted": 5, "sent": 4},
        "by_tenant": {"acme": {"attempted": 4}, "globex": {"attempted": 1}},
        "by_tenant_last_at": {"acme": OLD, "globex": FRESH},
    }
    row.update(overrides)
    return row


def test_record_worker_status_stamps_per_tenant_activity():
    service = _service()
    service._record_worker_status(
        "dispatch_retries", {"attempted": 2}, {"acme": {"attempted": 2}}
    )
    service._record_worker_status(
        "dispatch_retries", {"attempted": 1}, {"globex": {"attempted": 1}}
    )
    row = service._get_store().worker_status("dispatch_retries")
    # Each tenant's stamp is the last sweep that recorded an
    # outcome FOR IT — the second sweep does not refresh acme.
    assert set(row["by_tenant_last_at"]) == {"acme", "globex"}
    assert row["by_tenant_last_at"]["globex"] >= row["by_tenant_last_at"]["acme"]


def test_tenant_staleness_splits_by_tenant(monkeypatch):
    monkeypatch.setenv("WORKER_STALE_SECONDS", "3600")  # 1h
    view = worker_tenant_staleness({"dispatch_retries": _row()}, now=NOW)
    tenants = view["dispatch_retries"]
    assert tenants["acme"]["stale"] is True
    assert tenants["acme"]["age_seconds"] == 48 * 3600.0
    assert tenants["globex"]["stale"] is False
    # An unwatched worker claims no per-tenant verdict either.
    monkeypatch.delenv("WORKER_STALE_SECONDS")
    assert worker_tenant_staleness({"dispatch_retries": _row()}, now=NOW) == {}


def test_workers_view_and_endpoint(monkeypatch):
    monkeypatch.setenv("WORKER_STALE_SECONDS_DISPATCH_RETRIES", "3600")
    service = _service()
    store = service._get_store()
    row = _row(stale_episode={"detected_at": OLD, "outcome": "sent"})
    store.save_worker_status("dispatch_retries", row)
    store.save_worker_status("sla_sweep", {"worker": "sla_sweep", "sweeps": 1})

    view = service.workers_view(now=NOW)
    workers = {w["worker"]: w for w in view["workers"]}
    retries = workers["dispatch_retries"]
    assert retries["sweeps"] == 3
    assert retries["staleness"]["stale"] is False  # globally fresh
    assert retries["tenants"]["acme"]["stale"] is True  # …but not for acme
    assert retries["open_stale_episode"]["detected_at"] == OLD
    # sla_sweep is unwatched: no verdict is claimed for it.
    assert workers["sla_sweep"]["staleness"] is None
    assert workers["sla_sweep"]["tenants"] == {}

    monkeypatch.setattr(api_module, "service", service)
    client = TestClient(api_module.app)
    response = client.get("/workers")
    assert response.status_code == 200
    body = response.json()
    assert [w["worker"] for w in body["workers"]] == [
        "dispatch_retries",
        "sla_sweep",
    ]


def test_metrics_carry_the_per_tenant_worker_families(monkeypatch):
    monkeypatch.setenv("WORKER_STALE_SECONDS_DISPATCH_RETRIES", "3600")
    service = _service()
    # /metrics reads the real clock, so this row's stamps are
    # relative to it (the view tests above inject their own now).
    from datetime import timedelta

    real_now = datetime.now(timezone.utc)
    row = _row(
        last_sweep_at=(real_now - timedelta(minutes=30)).isoformat(),
        by_tenant_last_at={
            "acme": (real_now - timedelta(hours=48)).isoformat(),
            "globex": (real_now - timedelta(minutes=30)).isoformat(),
        },
    )
    service._get_store().save_worker_status("dispatch_retries", row)
    monkeypatch.setattr(api_module, "service", service)
    client = TestClient(api_module.app)
    text = client.get("/metrics").text
    assert (
        'shipment_agent_worker_tenant_stale{worker="dispatch_retries",tenant="acme"} 1'
        in text
    )
    assert (
        'shipment_agent_worker_tenant_stale{worker="dispatch_retries",tenant="globex"} 0'
        in text
    )


def test_digest_stale_workers_name_their_stale_tenants(monkeypatch):
    monkeypatch.setenv("WORKER_STALE_SECONDS_DISPATCH_RETRIES", "3600")
    service = _service()
    # Globally stale too (last sweep 48h old), with acme the tenant
    # whose work went quiet.
    service._get_store().save_worker_status(
        "dispatch_retries", _row(last_sweep_at=OLD)
    )
    digest = service.compose_queue_digest(now=NOW)
    (entry,) = digest["stale_workers"]
    assert entry["worker"] == "dispatch_retries"
    assert entry["stale_tenants"] == [
        {"tenant_id": "acme", "last_active_at": OLD, "age_seconds": 48 * 3600.0}
    ]
