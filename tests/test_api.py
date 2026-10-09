from fastapi.testclient import TestClient

from shipment_agent.api import app

client = TestClient(app)

SHIPMENT = {
    "shipment_id": "API-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T09:00:00",
    "latest_event": "Delayed at hub",
    "documents": [],
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_then_approve_flow():
    response = client.post("/shipments/analyze", json=SHIPMENT)
    assert response.status_code == 200
    body = response.json()
    assert body["classification"]["exception_type"] == "delay"
    assert body["approval_status"] == "awaiting_approval"
    assert body["external_action_taken"] is False

    approved = client.post("/shipments/API-1/approve", json={"approver": "ops-lead"})
    assert approved.status_code == 200
    assert approved.json()["approval_status"] == "approved"
    # Approval in the prototype still performs no external action.
    assert approved.json()["external_action_taken"] is False


def test_approve_unknown_shipment_404():
    response = client.post("/shipments/NOPE/approve", json={"approver": "x"})
    assert response.status_code == 404


def test_double_decision_is_422_in_both_directions():
    shipment = dict(SHIPMENT, shipment_id="API-2")
    assert client.post("/shipments/analyze", json=shipment).status_code == 200
    rejected = client.post(
        "/shipments/API-2/reject", json={"reviewer": "ops-lead", "reason": "call first"}
    )
    assert rejected.status_code == 200
    # A rejected shipment must not become approvable afterwards.
    approve_after_reject = client.post(
        "/shipments/API-2/approve", json={"approver": "ops-lead"}
    )
    assert approve_after_reject.status_code == 422

    shipment3 = dict(SHIPMENT, shipment_id="API-3")
    assert client.post("/shipments/analyze", json=shipment3).status_code == 200
    assert client.post("/shipments/API-3/approve", json={"approver": "a"}).status_code == 200
    # Nor can an approved shipment be approved a second time.
    approve_again = client.post("/shipments/API-3/approve", json={"approver": "b"})
    assert approve_again.status_code == 422


def test_approve_blocked_when_guardrails_fail():
    # The customer name lands verbatim in the draft greeting; a name that
    # contains a banned promise phrase makes the draft fail validation.
    shipment = dict(
        SHIPMENT, shipment_id="API-4", customer_name="We Guarantee Logistics LLC"
    )
    analyzed = client.post("/shipments/analyze", json=shipment)
    assert analyzed.status_code == 200
    assert analyzed.json()["validation"]["passed"] is False
    blocked = client.post("/shipments/API-4/approve", json={"approver": "ops-lead"})
    assert blocked.status_code == 422
    # And it stays undecided — rejection is still possible.
    rejected = client.post(
        "/shipments/API-4/reject", json={"reviewer": "ops-lead", "reason": "bad draft"}
    )
    assert rejected.status_code == 200
