"""API-key auth tests: with API_KEY set, data endpoints (read +
mutating) require the X-API-Key header; without it, the API is open
(the documented local-dev behaviour). Console page + health stay open.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shipment_agent.api import app

client = TestClient(app)

SHIPMENT = {
    "shipment_id": "AUTH-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

KEY = {"X-API-Key": "test-secret"}
WRONG = {"X-API-Key": "not-the-key"}


@pytest.fixture(autouse=True)
def no_key_by_default(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)


def test_open_when_api_key_unset():
    assert client.get("/samples").status_code == 200
    assert client.post("/shipments/analyze", json=SHIPMENT).status_code == 200


def test_read_endpoints_require_key_when_set(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")
    assert client.get("/samples").status_code == 401
    assert client.get("/policies").status_code == 401
    assert client.get("/evals/results").status_code == 401
    assert client.get("/shipments/AUTH-1").status_code == 401
    assert client.get("/samples", headers=WRONG).status_code == 401
    assert client.get("/samples", headers=KEY).status_code == 200
    assert client.get("/policies", headers=KEY).status_code == 200


def test_mutating_endpoints_require_key_when_set(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")
    assert client.post("/shipments/analyze", json=SHIPMENT).status_code == 401
    assert client.post("/shipments/analyze", json=SHIPMENT, headers=KEY).status_code == 200
    assert client.post(
        "/shipments/AUTH-1/approve", json={"approver": "ops"}
    ).status_code == 401
    approved = client.post(
        "/shipments/AUTH-1/approve", json={"approver": "ops"}, headers=KEY
    )
    assert approved.status_code == 200
    assert approved.json()["approval_status"] == "approved"
    rejected = client.post(
        "/shipments/AUTH-1/reject", json={"reviewer": "ops"}, headers=KEY
    )
    assert rejected.status_code == 422  # already decided — gate, not auth


def test_console_and_health_stay_open_when_key_set(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    # The console carries the key field so it keeps working gated.
    assert 'id="apikey"' in client.get("/").text
