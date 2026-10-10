"""Per-tenant key rotation: the outgoing key works until its grace
deadline, then stops — and every request says which generation
authenticated it.

Rotating a tenant's key used to be a hard cutover. Now the tenant
holds a current key plus the outgoing *previous* one
(``TENANT_PREVIOUS_API_KEYS`` / ``API_KEY_PREVIOUS_<TENANT>``), the
previous one authenticating until ``TENANT_KEY_ROTATED_AT`` +
``TENANT_KEY_GRACE_HOURS``. These tests pin the window arithmetic at
the config layer, the API outcomes (previous key works inside the
window, earns a precise 401 after it, a wrong-tenant previous key is
still a 403), the key id stamped on the record and carried into the
audit export, and the operator surfaces (``GET /auth/rotation`` and
the /metrics rotation families) — key ids and booleans only, never
secrets.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.config import (
    key_grace_deadline,
    tenant_key_generation,
    tenant_key_rotated_at,
    tenant_previous_api_key,
    tenant_rotation_status,
)
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

SHIPMENT = {
    "shipment_id": "ROT-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "carrier": "Synthetic Carrier",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

_KEY_VARS = (
    "API_KEY",
    "TENANT_API_KEYS",
    "TENANT_PREVIOUS_API_KEYS",
    "TENANT_KEY_ROTATED_AT",
    "TENANT_KEY_GRACE_HOURS",
    "TENANT_ID",
    "API_KEY_ACME",
    "API_KEY_PREVIOUS_ACME",
    "TENANT_KEY_ROTATED_AT_ACME",
    "TENANT_KEY_GRACE_HOURS_ACME",
)

ROTATED_AT = "2026-10-01T00:00:00+00:00"
IN_WINDOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)  # +36h
PAST_WINDOW = datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc)  # +120h


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


def _samples(client, key=None, tenant=None):
    headers = {}
    if key is not None:
        headers["X-API-Key"] = key
    if tenant is not None:
        headers["X-Tenant-ID"] = tenant
    return client.get("/samples", headers=headers)


# ---------------------------------------------------------------------------
# Configuration: the window arithmetic
# ---------------------------------------------------------------------------


def test_previous_key_sources_and_precedence(monkeypatch):
    monkeypatch.setenv("API_KEY_PREVIOUS_ACME", "var-key")
    assert tenant_previous_api_key("acme") == "var-key"
    assert tenant_previous_api_key("globex") is None
    # The map wins over the per-tenant variable, like the current key.
    monkeypatch.setenv("TENANT_PREVIOUS_API_KEYS", "acme:map-key")
    assert tenant_previous_api_key("acme") == "map-key"


def test_rotation_timestamp_and_deadline(monkeypatch, rotating_env):
    assert tenant_key_rotated_at("acme") == datetime(
        2026, 10, 1, tzinfo=timezone.utc
    )
    # Default grace: 72 hours.
    assert key_grace_deadline("acme") == datetime(
        2026, 10, 4, tzinfo=timezone.utc
    )
    # A per-tenant grace override wins over the global default.
    monkeypatch.setenv("TENANT_KEY_GRACE_HOURS_ACME", "24")
    assert key_grace_deadline("acme") == datetime(
        2026, 10, 2, tzinfo=timezone.utc
    )
    # No rotation recorded for globex: no deadline, no window.
    assert tenant_key_rotated_at("globex") is None
    assert key_grace_deadline("globex") is None


def test_unparseable_rotation_timestamp_opens_no_window(monkeypatch):
    monkeypatch.setenv("TENANT_PREVIOUS_API_KEYS", "acme:old-key")
    monkeypatch.setenv("TENANT_KEY_ROTATED_AT", "acme:not-a-date")
    assert tenant_key_rotated_at("acme") is None
    assert tenant_key_generation("acme", "old-key", now=IN_WINDOW) is None


def test_generation_inside_and_outside_the_window(monkeypatch, rotating_env):
    assert tenant_key_generation("acme", "new-key", now=IN_WINDOW) == "current"
    assert tenant_key_generation("acme", "old-key", now=IN_WINDOW) == "previous"
    # At the deadline itself the window is still open; past it, closed.
    deadline = datetime(2026, 10, 4, tzinfo=timezone.utc)
    assert tenant_key_generation("acme", "old-key", now=deadline) == "previous"
    assert (
        tenant_key_generation("acme", "old-key", now=deadline + timedelta(seconds=1))
        is None
    )
    assert tenant_key_generation("acme", "old-key", now=PAST_WINDOW) is None
    assert tenant_key_generation("acme", "other", now=IN_WINDOW) is None
    assert tenant_key_generation("acme", None, now=IN_WINDOW) is None


def test_previous_key_without_a_rotation_timestamp_never_validates(monkeypatch):
    monkeypatch.setenv("TENANT_PREVIOUS_API_KEYS", "acme:old-key")
    assert tenant_key_generation("acme", "old-key", now=IN_WINDOW) is None


def test_rotation_status_reports_facts_never_secrets(monkeypatch, rotating_env):
    status = tenant_rotation_status("acme", now=IN_WINDOW)
    assert status["current_key_configured"] is True
    assert status["previous_key_configured"] is True
    assert status["grace_open"] is True
    assert status["grace_deadline"] == "2026-10-04T00:00:00+00:00"
    assert "new-key" not in str(status) and "old-key" not in str(status)
    assert tenant_rotation_status("acme", now=PAST_WINDOW)["grace_open"] is False


# ---------------------------------------------------------------------------
# The API: generations at the gate
# ---------------------------------------------------------------------------


def test_previous_key_authenticates_inside_the_window(client, monkeypatch, rotating_env):
    api_client, _ = client
    _at(monkeypatch, IN_WINDOW)
    assert _samples(api_client, key="old-key", tenant="acme").status_code == 200
    assert _samples(api_client, key="new-key", tenant="acme").status_code == 200


def test_previous_key_stops_at_the_deadline(client, monkeypatch, rotating_env):
    api_client, _ = client
    _at(monkeypatch, PAST_WINDOW)
    response = _samples(api_client, key="old-key", tenant="acme")
    assert response.status_code == 401
    assert "grace window has closed" in response.json()["detail"]
    # The current key is unaffected by the window closing.
    assert _samples(api_client, key="new-key", tenant="acme").status_code == 200


def test_previous_key_aimed_at_another_tenant_is_a_403(client, monkeypatch, rotating_env):
    api_client, _ = client
    _at(monkeypatch, IN_WINDOW)
    response = _samples(api_client, key="old-key", tenant="globex")
    assert response.status_code == 403


def test_the_analysis_records_which_generation_authenticated_it(
    client, monkeypatch, rotating_env
):
    api_client, service = client
    _at(monkeypatch, IN_WINDOW)
    headers = {"X-API-Key": "old-key", "X-Tenant-ID": "acme"}
    response = api_client.post("/shipments/analyze", json=SHIPMENT, headers=headers)
    assert response.status_code == 200
    record = service._get_store().get("ROT-1", tenant_id="acme")
    assert record.auth_key_id == "acme:previous"
    # And the audit export carries the key id (never the secret).
    export = api_client.get("/audit/export", headers=headers)
    assert export.status_code == 200
    rows = export.json()["decisions"]
    assert rows[0]["api_key_id"] == "acme:previous"
    assert "old-key" not in str(export.json())


def test_current_key_records_its_own_generation(client, monkeypatch, rotating_env):
    api_client, service = client
    _at(monkeypatch, IN_WINDOW)
    headers = {"X-API-Key": "new-key", "X-Tenant-ID": "acme"}
    response = api_client.post("/shipments/analyze", json=SHIPMENT, headers=headers)
    assert response.status_code == 200
    record = service._get_store().get("ROT-1", tenant_id="acme")
    assert record.auth_key_id == "acme:current"


def test_shared_key_records_the_shared_id(client, monkeypatch):
    api_client, service = client
    monkeypatch.setenv("API_KEY", "the-shared-key")
    response = api_client.post(
        "/shipments/analyze", json=SHIPMENT, headers={"X-API-Key": "the-shared-key"}
    )
    assert response.status_code == 200
    record = service._get_store().get("ROT-1", tenant_id="default")
    assert record.auth_key_id == "shared"


# ---------------------------------------------------------------------------
# The operator surfaces
# ---------------------------------------------------------------------------


def test_rotation_operator_view_counts_previous_key_use(
    client, monkeypatch, rotating_env
):
    api_client, _ = client
    _at(monkeypatch, IN_WINDOW)
    headers = {"X-API-Key": "old-key", "X-Tenant-ID": "acme"}
    api_client.post("/shipments/analyze", json=SHIPMENT, headers=headers)
    view = api_client.get("/auth/rotation", headers=headers)
    assert view.status_code == 200
    body = view.json()
    assert body["tenant_id"] == "acme"
    assert body["grace_open"] is True
    assert body["previous_key_requests"]["count"] == 1
    assert body["previous_key_requests"]["last_at"]
    assert "old-key" not in view.text and "new-key" not in view.text


def test_metrics_carry_the_rotation_families(client, monkeypatch, rotating_env):
    api_client, _ = client
    _at(monkeypatch, IN_WINDOW)
    headers = {"X-API-Key": "old-key", "X-Tenant-ID": "acme"}
    api_client.post("/shipments/analyze", json=SHIPMENT, headers=headers)
    text = api_client.get("/metrics").text
    assert 'shipment_agent_tenant_previous_key_requests_total{tenant="acme"} 1' in text
    assert 'shipment_agent_tenant_key_grace_open{tenant="acme"} 1' in text
    assert "old-key" not in text and "new-key" not in text


def test_auth_key_id_survives_the_sqlite_double(tmp_path):
    from shipment_agent.store import SQLiteStore

    store = SQLiteStore(tmp_path / "state.db")
    service = ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store,
        checkpointer=False,
    )
    service.analyze(SHIPMENT, tenant_id="acme", auth_key_id="acme:previous")
    record = store.get("ROT-1", tenant_id="acme")
    assert record.auth_key_id == "acme:previous"
