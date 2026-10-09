"""Decision-actor tests: one name (`actor`) for approve AND reject.

Approve took `approver`, reject took `reviewer` — a client integrating
both endpoints had to remember two names for the same concept. Both now
accept `actor` (the legacy names stay as aliases), and the decision is
returned on the result as `decided_by`.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from shipment_agent.api import app
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import SQLiteStore

client = TestClient(app)

SHIPMENT = {
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


def _analyze(shipment_id: str) -> None:
    response = client.post(
        "/shipments/analyze", json={**SHIPMENT, "shipment_id": shipment_id}
    )
    assert response.status_code == 200


def test_approve_accepts_actor_and_returns_decided_by():
    _analyze("ACT-1")
    response = client.post("/shipments/ACT-1/approve", json={"actor": "qa-lead"})
    assert response.status_code == 200
    assert response.json()["approval_status"] == "approved"
    assert response.json()["decided_by"] == "qa-lead"


def test_reject_accepts_actor_and_returns_decided_by():
    _analyze("ACT-2")
    response = client.post(
        "/shipments/ACT-2/reject", json={"actor": "qa-lead", "reason": "call first"}
    )
    assert response.status_code == 200
    assert response.json()["approval_status"] == "rejected"
    assert response.json()["decided_by"] == "qa-lead"


def test_legacy_names_still_work_and_set_decided_by():
    _analyze("ACT-3")
    response = client.post("/shipments/ACT-3/approve", json={"approver": "ops-lead"})
    assert response.status_code == 200
    assert response.json()["decided_by"] == "ops-lead"
    _analyze("ACT-4")
    response = client.post(
        "/shipments/ACT-4/reject", json={"reviewer": "ops-lead", "reason": "no"}
    )
    assert response.status_code == 200
    assert response.json()["decided_by"] == "ops-lead"


def test_missing_actor_is_a_clean_422():
    _analyze("ACT-5")
    response = client.post("/shipments/ACT-5/approve", json={})
    assert response.status_code == 422
    assert "actor" in response.json()["detail"]


def test_decided_by_persists_in_the_store(tmp_path):
    service = ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=SQLiteStore(tmp_path / "state.db"),
    )
    service.analyze({**SHIPMENT, "shipment_id": "ACT-6"})
    service.approve("ACT-6", approver="qa-lead")
    stored = service.get("ACT-6")
    assert stored is not None
    assert stored.decided_by == "qa-lead"
