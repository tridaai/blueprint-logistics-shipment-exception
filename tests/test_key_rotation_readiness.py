"""Key-rotation completion assist: a grace window that has gone
quiet is *reported* ready to close — never closed for you.

The rotation endgame used to be manual: watch /auth/rotation,
notice the old key stopped being used, then unset it.
``rotation_close_readiness`` computes the notice from the recorded
evidence — the window is ready when it is still open and the
previous key has recorded no use for ``TENANT_KEY_QUIET_HOURS``
(the quiet stretch runs from the later of the rotation and the
last recorded previous-key request). These tests pin the
arithmetic at the config layer, then the three surfaces that speak
it: GET /auth/rotation, the /metrics closing family, and the
escalation digest's open-windows section. Retiring the key remains
a human's configuration change throughout — nothing here unsets
anything.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.config import rotation_close_readiness, tenant_key_quiet_hours
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

ROTATED_AT = "2026-10-01T00:00:00+00:00"  # grace default 72h -> deadline 10-04
IN_WINDOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)  # +36h
PAST_WINDOW = datetime(2026, 10, 6, 0, 0, tzinfo=timezone.utc)  # +120h

_KEY_VARS = (
    "API_KEY",
    "TENANT_API_KEYS",
    "TENANT_PREVIOUS_API_KEYS",
    "TENANT_KEY_ROTATED_AT",
    "TENANT_KEY_GRACE_HOURS",
    "TENANT_KEY_QUIET_HOURS",
    "TENANT_KEY_QUIET_HOURS_ACME",
    "TENANT_ID",
)


@pytest.fixture(autouse=True)
def clean_key_env(monkeypatch):
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.service.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.api.load_dotenv", lambda *a, **k: None)


@pytest.fixture()
def rotating_env(monkeypatch):
    monkeypatch.setenv("TENANT_API_KEYS", "acme:new-key,globex:globex-key")
    monkeypatch.setenv("TENANT_PREVIOUS_API_KEYS", "acme:old-key")
    monkeypatch.setenv("TENANT_KEY_ROTATED_AT", f"acme:{ROTATED_AT}")


def test_quiet_hours_precedence_and_clamp(monkeypatch, rotating_env):
    assert tenant_key_quiet_hours("acme") == 24.0  # the default
    monkeypatch.setenv("TENANT_KEY_QUIET_HOURS", "10")
    assert tenant_key_quiet_hours("acme") == 10.0
    monkeypatch.setenv("TENANT_KEY_QUIET_HOURS_ACME", "2.5")
    assert tenant_key_quiet_hours("acme") == 2.5  # per-tenant wins
    monkeypatch.setenv("TENANT_KEY_QUIET_HOURS_ACME", "-4")
    assert tenant_key_quiet_hours("acme") == 0.0  # clamped, not an error


def test_readiness_needs_a_window_and_an_open_one(monkeypatch):
    # No previous key at all.
    verdict = rotation_close_readiness("acme", now=IN_WINDOW)
    assert verdict["ready_to_close"] is False
    # Previous key but no recorded rotation: no window exists.
    monkeypatch.setenv("TENANT_PREVIOUS_API_KEYS", "acme:old-key")
    verdict = rotation_close_readiness("acme", now=IN_WINDOW)
    assert verdict["ready_to_close"] is False
    assert verdict["quiet_since"] is None


def test_readiness_never_used_key_quiet_since_rotation(
    monkeypatch, rotating_env
):
    verdict = rotation_close_readiness("acme", now=IN_WINDOW)
    # 36h of quiet inside a 72h window, quiet budget 24h: ready.
    assert verdict["ready_to_close"] is True
    assert verdict["quiet_since"] == ROTATED_AT
    assert verdict["last_previous_key_request_at"] is None
    # One hour after the rotation, the same silence is not yet a
    # 24h quiet stretch.
    early = datetime(2026, 10, 1, 1, 0, tzinfo=timezone.utc)
    assert rotation_close_readiness("acme", now=early)["ready_to_close"] is False
    # Past the deadline the window is closed already — nothing to
    # be ready about.
    verdict = rotation_close_readiness("acme", now=PAST_WINDOW)
    assert verdict["ready_to_close"] is False


def test_readiness_quiet_stretch_runs_from_the_last_use(
    monkeypatch, rotating_env
):
    last_use = "2026-10-02T06:00:00+00:00"  # 6h before IN_WINDOW
    verdict = rotation_close_readiness(
        "acme", last_previous_use_at=last_use, now=IN_WINDOW
    )
    assert verdict["ready_to_close"] is False
    assert verdict["quiet_since"] == last_use
    assert verdict["last_previous_key_request_at"] == last_use
    # A shorter configured quiet budget reads the same evidence
    # the other way.
    monkeypatch.setenv("TENANT_KEY_QUIET_HOURS", "3")
    verdict = rotation_close_readiness(
        "acme", last_previous_use_at=last_use, now=IN_WINDOW
    )
    assert verdict["ready_to_close"] is True


SHIPMENT = {
    "shipment_id": "ROT-9",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "carrier": "Synthetic Carrier",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


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


def _use_previous_key(api_client, service, at_iso):
    headers = {"X-API-Key": "old-key", "X-Tenant-ID": "acme"}
    response = api_client.post("/shipments/analyze", json=SHIPMENT, headers=headers)
    assert response.status_code == 200
    record = service._get_store().get("ROT-9", tenant_id="acme")
    record.created_at = at_iso  # the recorded use, at a controlled time
    service._get_store().save(record)


def test_rotation_view_and_metrics_speak_readiness(
    client, monkeypatch, rotating_env
):
    api_client, service = client
    monkeypatch.setattr(api_module, "_auth_now", lambda: IN_WINDOW)
    headers = {"X-API-Key": "new-key", "X-Tenant-ID": "acme"}

    # No recorded use: quiet since the rotation — ready.
    body = api_client.get("/auth/rotation", headers=headers).json()
    assert body["ready_to_close"] is True
    assert body["quiet_hours"] == 24.0
    assert body["quiet_since"] == ROTATED_AT
    text = api_client.get("/metrics").text
    assert 'shipment_agent_tenant_key_rotation_closing{tenant="acme"} 1' in text

    # A use six hours ago restarts the quiet stretch: not ready.
    _use_previous_key(api_client, service, "2026-10-02T06:00:00+00:00")
    body = api_client.get("/auth/rotation", headers=headers).json()
    assert body["ready_to_close"] is False
    assert body["quiet_since"] == "2026-10-02T06:00:00+00:00"
    assert body["last_previous_key_request_at"] == "2026-10-02T06:00:00+00:00"
    text = api_client.get("/metrics").text
    assert 'shipment_agent_tenant_key_rotation_closing{tenant="acme"} 0' in text
    # And the previous key still authenticates — readiness changed
    # nothing about the window itself.
    assert body["grace_open"] is True


def test_digest_open_windows_name_readiness(client, monkeypatch, rotating_env):
    api_client, service = client
    # The previous key only authenticates inside its window, so the
    # recorded use is made with the API clock pinned inside it.
    monkeypatch.setattr(
        api_module,
        "_auth_now",
        lambda: datetime(2026, 10, 1, 2, 0, tzinfo=timezone.utc),
    )
    _use_previous_key(api_client, service, "2026-10-01T02:00:00+00:00")
    digest = service.compose_queue_digest(now=IN_WINDOW)
    (window,) = digest["open_key_rotation_windows"]
    assert window["tenant_id"] == "acme"
    # Last use 34h before the digest moment: quiet past the 24h
    # budget, so the digest names the window ready to close.
    assert window["ready_to_close"] is True
    assert window["last_previous_key_request_at"] == "2026-10-01T02:00:00+00:00"
