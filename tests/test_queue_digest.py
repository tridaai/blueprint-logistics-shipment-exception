"""The escalation digest: the queue's morning picture, composed
by the sweep into a stored summary row and pulled by the API.

The ladder pages per case; the digest is the shift-level view —
breaches by stage, the last 24h of firings (escalations naming the
per-severity factor that fired), the oldest waiter per severity
per tenant, stale workers, and open key-rotation windows. All of
it composed in code from the existing projections, so this file
pins the composition against shaped records, the sweep's storing
of it, and the endpoint's pull-first contract.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import load_sample_shipments
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore, SQLiteStore

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


def _service(store=None) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store or InMemoryStore(),
        checkpointer=False,
    )


def _sample(shipment_id: str) -> dict:
    return next(
        s for s in load_sample_shipments() if s["shipment_id"] == shipment_id
    )


def _aged(service: ShipmentService, shipment_id: str, hours: float, tenant_id=None):
    """Analyse a sample, then pin its record's age relative to NOW."""
    kwargs = {"tenant_id": tenant_id} if tenant_id else {}
    service.analyze(_sample(shipment_id), **kwargs)
    store = service._get_store()
    record = store.get(shipment_id, tenant_id=tenant_id or "default")
    record.created_at = (NOW - timedelta(hours=hours)).isoformat()
    store.save(record)
    return record


def _digest_fixture(service: ShipmentService):
    """Three awaiting records across two tenants and three stages.

    SYN-1001 (high: 24h budget, 48h escalation) at 100h old in the
    default tenant — escalated, with both ladder rungs fired inside
    the window. SYN-1006 (critical: 4h/8h) at 30h old for acme —
    escalated too, but its breach fired outside the window.
    SYN-1004 (medium: 48h/96h) at 50h old for acme — fresh breach.
    """
    old = _aged(service, "SYN-1001", 100)
    old.sla_dispatch_attempts = [
        {
            "attempt": 1,
            "at": (NOW - timedelta(hours=2)).isoformat(),
            "outcome": "sent",
            "http_status": 200,
            "signature_id": "sha256=x",
            "error": None,
        }
    ]
    old.sla_escalation_attempts = [
        {
            "attempt": 1,
            "at": (NOW - timedelta(hours=1)).isoformat(),
            "outcome": "sent",
            "http_status": 200,
            "signature_id": "sha256=y",
            "error": None,
        }
    ]
    service._get_store().save(old)

    acme_critical = _aged(service, "SYN-1006", 30, tenant_id="acme")
    acme_critical.sla_dispatch_attempts = [
        {
            "attempt": 1,
            "at": (NOW - timedelta(hours=30)).isoformat(),
            "outcome": "failed",
            "http_status": 500,
            "signature_id": None,
            "error": "HTTP 500",
        }
    ]
    service._get_store().save(acme_critical)

    _aged(service, "SYN-1004", 50, tenant_id="acme")
    return service


def test_digest_composition_counts_stages_and_firings():
    service = _digest_fixture(_service())
    digest = service.compose_queue_digest(now=NOW)

    assert digest["awaiting_total"] == 3
    assert digest["breaches_by_stage"] == {
        "within_budget": 0,
        "breach": 1,
        "escalated": 2,
    }
    assert digest["breaches_by_severity"] == {
        "critical": 1,
        "high": 1,
        "medium": 1,
    }
    # Only the firing inside the 24h window is listed; the acme
    # breach fired 30h ago and stays history, not news.
    (breach,) = digest["breaches_fired"]
    assert breach["shipment_id"] == "SYN-1001"
    assert breach["tenant_id"] == "default"
    assert breach["severity"] == "high"
    assert breach["outcome"] == "sent"
    (escalation,) = digest["escalations_fired"]
    assert escalation["shipment_id"] == "SYN-1001"
    assert escalation["escalation_factor"] == 2.0  # the high factor that fired
    assert escalation["at"] == (NOW - timedelta(hours=1)).isoformat()


def test_digest_names_the_oldest_waiter_per_severity_per_tenant():
    service = _digest_fixture(_service())
    # A second, younger high-severity waiter for acme must not
    # displace the critical — and a younger default-tenant high
    # must not displace SYN-1001.
    _aged(service, "SYN-1001", 10, tenant_id="acme")
    digest = service.compose_queue_digest(now=NOW)
    oldest = digest["oldest_waiter_per_severity"]
    assert oldest["default"]["high"]["shipment_id"] == "SYN-1001"
    assert oldest["default"]["high"]["age_seconds"] == 100 * 3600
    assert oldest["acme"]["critical"]["shipment_id"] == "SYN-1006"
    assert oldest["acme"]["medium"]["shipment_id"] == "SYN-1004"
    assert oldest["acme"]["high"]["age_seconds"] == 10 * 3600


def test_digest_reports_stale_workers_and_open_rotation_windows(monkeypatch):
    service = _digest_fixture(_service())
    store = service._get_store()
    store.save_worker_status(
        "sla_sweep",
        {
            "worker": "sla_sweep",
            "last_sweep_at": (NOW - timedelta(hours=2)).isoformat(),
            "updated_at": (NOW - timedelta(hours=2)).isoformat(),
        },
    )
    monkeypatch.setenv("WORKER_STALE_SECONDS", "3600")
    monkeypatch.setenv("TENANT_API_KEYS", "acme:current-key,globex:g-key")
    monkeypatch.setenv("TENANT_PREVIOUS_API_KEYS", "acme:previous-key")
    monkeypatch.setenv(
        "TENANT_KEY_ROTATED_AT", f"acme:{(NOW - timedelta(hours=1)).isoformat()}"
    )
    digest = service.compose_queue_digest(now=NOW)

    (stale,) = digest["stale_workers"]
    assert stale["worker"] == "sla_sweep"
    assert stale["age_seconds"] == 2 * 3600
    assert stale["threshold_seconds"] == 3600.0
    # acme rotated an hour ago with a 72h grace: the window is open.
    # globex holds no previous key: no window exists for it at all.
    (window,) = digest["open_key_rotation_windows"]
    assert window["tenant_id"] == "acme"
    assert window["grace_hours"] == 72.0
    assert window["grace_deadline"] is not None


def test_digest_is_empty_but_well_formed_on_an_empty_store():
    digest = _service().compose_queue_digest(now=NOW)
    assert digest["awaiting_total"] == 0
    assert digest["breaches_by_stage"] == {
        "within_budget": 0,
        "breach": 0,
        "escalated": 0,
    }
    assert digest["breaches_fired"] == []
    assert digest["escalations_fired"] == []
    assert digest["oldest_waiter_per_severity"] == {}
    assert digest["stale_workers"] == []
    assert digest["open_key_rotation_windows"] == []


def test_pull_first_composes_on_read_until_a_sweep_stores_the_digest():
    service = _digest_fixture(_service())
    before = service.queue_digest(now=NOW)
    assert before["stored"] is False
    assert before["awaiting_total"] == 3

    service.sla_breach_sweep(now=NOW)
    stored = service._get_store().summary("queue_digest")
    assert stored is not None
    assert stored["awaiting_total"] == 3

    after = service.queue_digest()
    assert after["stored"] is True
    assert after["generated_at"] == stored["generated_at"]


@pytest.fixture(params=["memory", "sqlite"])
def summary_store(request, tmp_path):
    if request.param == "memory":
        return InMemoryStore()
    return SQLiteStore(tmp_path / "approvals.db")


def test_summary_rows_round_trip_on_the_store_doubles(summary_store):
    assert summary_store.summary("queue_digest") is None
    summary_store.save_summary("queue_digest", {"generated_at": "t", "n": 1})
    assert summary_store.summary("queue_digest") == {"generated_at": "t", "n": 1}
    summary_store.save_summary("queue_digest", {"generated_at": "t2", "n": 2})
    assert summary_store.summary("queue_digest")["n"] == 2


def test_queue_digest_endpoint(monkeypatch):
    service = _digest_fixture(_service())
    service.sla_breach_sweep(now=NOW)  # the sweep stores the digest
    monkeypatch.setattr(api_module, "service", service)
    client = TestClient(api_module.app)
    body = client.get("/queue/digest").json()
    assert body["awaiting_total"] == 3
    assert body["stored"] is True  # pull-first: the sweep's row
    assert body["breaches_by_stage"]["escalated"] == 2
    assert body["escalations_fired"][0]["escalation_factor"] == 2.0
