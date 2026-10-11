"""Key-use evidence beyond analyses: the rotation's closing proof.

Rotation readiness used to read stored analyses only — an outgoing
key used solely for reads (the queue, the corpus) left no record,
so "the old key has gone quiet" could be declared while it was
still calling. Now the auth gate stamps every authenticated
request on a per-key counter (ids, counts, times, endpoint class —
never key material), and ``GET /auth/rotation``, /metrics, and the
digest all read the merged evidence: analyses ∪ counter, the
later last use winning. These tests pin the stamping (reads count,
current-key use never touches the previous counter, failed auth
stamps nothing), the window slice (a new rotation opens a fresh
count), the deadline boundary under the pinned API clock, and the
readiness correction itself.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore, SQLiteStore

_KEY_VARS = (
    "API_KEY",
    "TENANT_API_KEYS",
    "TENANT_PREVIOUS_API_KEYS",
    "TENANT_KEY_ROTATED_AT",
    "TENANT_KEY_GRACE_HOURS",
    "TENANT_KEY_QUIET_HOURS",
    "TENANT_ID",
    "API_KEY_ACME",
    "API_KEY_PREVIOUS_ACME",
    "TENANT_KEY_ROTATED_AT_ACME",
    "TENANT_KEY_GRACE_HOURS_ACME",
)

ROTATED_AT = "2026-10-01T00:00:00+00:00"
DEADLINE = datetime(2026, 10, 4, 0, 0, tzinfo=timezone.utc)  # rotated + 72h
IN_WINDOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)

OLD = {"X-API-Key": "old-key", "X-Tenant-ID": "acme"}
NEW = {"X-API-Key": "new-key", "X-Tenant-ID": "acme"}


@pytest.fixture(autouse=True)
def clean_key_env(monkeypatch):
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.service.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.api.load_dotenv", lambda *a, **k: None)


@pytest.fixture()
def rotating_env(monkeypatch):
    """Acme mid-rotation: new key current, old key in a 72h window."""
    monkeypatch.setenv("TENANT_API_KEYS", "acme:new-key,globex:globex-key")
    monkeypatch.setenv("TENANT_PREVIOUS_API_KEYS", "acme:old-key")
    monkeypatch.setenv("TENANT_KEY_ROTATED_AT", f"acme:{ROTATED_AT}")


@pytest.fixture()
def client(monkeypatch):
    service = ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=InMemoryStore(),
        checkpointer=False,
    )
    monkeypatch.setattr(api_module, "service", service)
    return TestClient(api_module.app), service


def _at(monkeypatch, moment):
    monkeypatch.setattr(api_module, "_auth_now", lambda: moment)


def _service(store=None) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store if store is not None else InMemoryStore(),
        checkpointer=False,
    )


def test_previous_key_reads_are_counted_not_just_analyses(
    client, monkeypatch, rotating_env
):
    api_client, service = client
    _at(monkeypatch, IN_WINDOW)
    assert api_client.get("/samples", headers=OLD).status_code == 200
    assert api_client.get("/queue", headers=OLD).status_code == 200

    counter = service.key_use_counter("acme", "acme:previous")
    assert counter["requests"] == 2
    assert counter["first_used_at"] == IN_WINDOW.isoformat()
    assert counter["last_used_at"] == IN_WINDOW.isoformat()
    assert counter["last_endpoint"] == "queue"
    assert counter["last_method"] == "GET"
    assert counter["window"] == {
        "rotated_at": ROTATED_AT,
        "requests": 2,
        "last_used_at": IN_WINDOW.isoformat(),
    }

    body = api_client.get("/auth/rotation", headers=NEW).json()
    # The analyses trail saw nothing — no record was stored…
    assert body["previous_key_requests"] == {"count": 0, "last_at": None}
    # …but the request trail saw both reads.
    assert body["previous_key_use"]["total"] == 2
    assert body["previous_key_use"]["in_window"] == 2
    assert body["previous_key_use"]["last_at"] == IN_WINDOW.isoformat()
    assert body["previous_key_use"]["last_endpoint"] == "queue"


def test_reads_reset_the_quiet_clock(client, monkeypatch, rotating_env):
    api_client, service = client
    read_at = datetime(2026, 10, 2, 0, 0, tzinfo=timezone.utc)
    _at(monkeypatch, read_at)
    assert api_client.get("/samples", headers=OLD).status_code == 200

    # At +25h from the read the quiet budget (24h) is met and the
    # window reads ready — with the evidence naming the read, not
    # an analysis that never happened.
    _at(monkeypatch, read_at + timedelta(hours=25))
    body = api_client.get("/auth/rotation", headers=NEW).json()
    assert body["ready_to_close"] is True
    assert body["quiet_since"] == read_at.isoformat()
    assert body["last_previous_key_request_at"] == read_at.isoformat()

    # One hour after the read, though, the same window is not
    # quiet at all — under analyses-only evidence this is the case
    # that used to be declared ready.
    _at(monkeypatch, read_at + timedelta(hours=1))
    body = api_client.get("/auth/rotation", headers=NEW).json()
    assert body["ready_to_close"] is False
    assert body["quiet_since"] == read_at.isoformat()


def test_window_opens_fresh_on_a_new_rotation(client, monkeypatch, rotating_env):
    api_client, service = client
    _at(monkeypatch, IN_WINDOW)
    api_client.get("/samples", headers=OLD)
    api_client.get("/samples", headers=OLD)
    assert service.key_use_counter("acme", "acme:previous")["requests"] == 2

    # A second rotation: the same outgoing key value, a new moment.
    # The next stamp opens a fresh window slice; the total keeps
    # counting, and the view reports the new window's count only.
    rotated_two = "2026-10-05T00:00:00+00:00"
    monkeypatch.setenv("TENANT_KEY_ROTATED_AT", f"acme:{rotated_two}")
    _at(monkeypatch, datetime(2026, 10, 5, 1, 0, tzinfo=timezone.utc))
    assert api_client.get("/samples", headers=OLD).status_code == 200
    counter = service.key_use_counter("acme", "acme:previous")
    assert counter["requests"] == 3
    assert counter["window"]["rotated_at"] == rotated_two
    assert counter["window"]["requests"] == 1
    body = api_client.get("/auth/rotation", headers=NEW).json()
    assert body["previous_key_use"]["total"] == 3
    assert body["previous_key_use"]["in_window"] == 1


def test_deadline_boundary_stamps_inside_and_never_outside(
    client, monkeypatch, rotating_env
):
    api_client, service = client
    # At the deadline itself the outgoing key still authenticates
    # (the generation check is at-or-before) — and the use counts.
    _at(monkeypatch, DEADLINE)
    assert api_client.get("/samples", headers=OLD).status_code == 200
    assert service.key_use_counter("acme", "acme:previous")["requests"] == 1

    # One second past it: the precise closed-window 401, and a
    # failed authentication stamps nothing.
    _at(monkeypatch, DEADLINE + timedelta(seconds=1))
    response = api_client.get("/samples", headers=OLD)
    assert response.status_code == 401
    assert "grace window has closed" in response.json()["detail"]
    assert service.key_use_counter("acme", "acme:previous")["requests"] == 1


def test_current_key_use_never_touches_the_previous_counter(
    client, monkeypatch, rotating_env
):
    api_client, service = client
    _at(monkeypatch, IN_WINDOW)
    api_client.get("/samples", headers=NEW)
    api_client.get("/queue", headers=NEW)
    assert service.key_use_counter("acme", "acme:previous") is None
    current = service.key_use_counter("acme", "acme:current")
    assert current["requests"] == 2
    assert current["window"]["rotated_at"] == ROTATED_AT


def test_counters_and_views_never_hold_key_material(
    client, monkeypatch, rotating_env
):
    api_client, service = client
    _at(monkeypatch, IN_WINDOW)
    api_client.get("/samples", headers=OLD)
    api_client.get("/samples", headers=NEW)
    surfaces = [
        json.dumps(service.key_use_counter("acme", "acme:previous")),
        json.dumps(service.key_use_counter("acme", "acme:current")),
        json.dumps(service.previous_key_use("acme", rotated_at=ROTATED_AT)),
        api_client.get("/auth/rotation", headers=NEW).text,
        api_client.get("/metrics").text,
    ]
    for surface in surfaces:
        assert "old-key" not in surface
        assert "new-key" not in surface


def test_metrics_report_window_requests_and_merged_readiness(
    client, monkeypatch, rotating_env
):
    api_client, service = client
    _at(monkeypatch, IN_WINDOW)
    api_client.get("/samples", headers=OLD)
    text = api_client.get("/metrics").text
    # The whole-API evidence: one authenticated request in window…
    assert (
        'shipment_agent_tenant_previous_key_authenticated_requests_total'
        '{tenant="acme"} 1'
    ) in text
    # …beside the analyses-only family, honestly still zero…
    assert 'shipment_agent_tenant_previous_key_requests_total{tenant="acme"} 0' in text
    # …and readiness on the scrape reads the read as use, not quiet.
    assert 'shipment_agent_tenant_key_rotation_closing{tenant="acme"} 0' in text


def test_counter_survives_the_sqlite_double(tmp_path):
    path = tmp_path / "keyuse.db"
    service = _service(SQLiteStore(path))
    service.record_key_use(
        "acme",
        "acme:previous",
        at="2026-10-02T12:00:00+00:00",
        method="GET",
        endpoint="queue",
        rotated_at=ROTATED_AT,
    )
    service.record_key_use(
        "acme",
        "acme:previous",
        at="2026-10-02T13:00:00+00:00",
        method="GET",
        endpoint="samples",
        rotated_at=ROTATED_AT,
    )
    reopened = _service(SQLiteStore(path))
    counter = reopened.key_use_counter("acme", "acme:previous")
    assert counter["requests"] == 2
    assert counter["last_used_at"] == "2026-10-02T13:00:00+00:00"
    use = reopened.previous_key_use("acme", rotated_at=ROTATED_AT)
    assert use["requests"]["in_window"] == 2
    assert use["last_used_at"] == "2026-10-02T13:00:00+00:00"


def test_shared_key_requests_are_counted_under_the_shared_id(
    client, monkeypatch
):
    api_client, service = client
    monkeypatch.setenv("API_KEY", "shared-secret")
    headers = {"X-API-Key": "shared-secret"}
    assert api_client.get("/samples", headers=headers).status_code == 200
    # Under the shared model the tenant header is a trusted claim:
    # the same key, another partition, its own counter row.
    claimed = {**headers, "X-Tenant-ID": "acme"}
    assert api_client.get("/samples", headers=claimed).status_code == 200
    assert service.key_use_counter("default", "shared")["requests"] == 1
    assert service.key_use_counter("acme", "shared")["requests"] == 1
    assert "shared-secret" not in json.dumps(
        service.key_use_counter("acme", "shared")
    )


def test_open_api_stamps_nothing(client):
    api_client, service = client
    # No key configured anywhere: nothing authenticated, so
    # nothing is recorded — an open local deployment accrues no
    # evidence it never earned.
    assert api_client.get("/samples").status_code == 200
    assert service.key_use_counter("default", "shared") is None
    assert service.previous_key_use("acme")["requests"]["total"] == 0
