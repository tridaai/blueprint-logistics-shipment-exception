"""Reviewer feedback loop tests: decisions teach the next diagnosis.

Approve/reject accept an optional ``reason``; both stores keep it.
When a later shipment is analysed for the same consignee or the same
lane, the most recent decisions (last 3, reasons truncated) surface in
the diagnosis evidence as "reviewer feedback: …" lines. The decided
shipment's own result echoes the reason. No reason, no feedback —
oversight only teaches when the decider said why.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from shipment_agent.api import app
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore, SQLiteStore


def _damage(
    shipment_id: str,
    customer: str = "Acme Parts",
    origin: str = "Memphis, TN",
    destination: str = "Charlotte, NC",
) -> dict:
    return {
        "shipment_id": shipment_id,
        "origin": origin,
        "destination": destination,
        "customer_name": customer,
        "latest_event": "Arrived at destination terminal",
        "condition_notes": "Two cartons crushed, contents leaking noted at terminal inspection",
        "documents": [],
    }


def _service(store=None) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store or InMemoryStore(),
    )


def _feedback_lines(result) -> list[str]:
    assert result.diagnosis is not None
    return [e for e in result.diagnosis.evidence if e.startswith("reviewer feedback:")]


# --------------------------------------------------------------------------
# The loop: a reasoned decision surfaces on the next matching case
# --------------------------------------------------------------------------

def test_reject_reason_surfaces_on_the_next_same_lane_case():
    service = _service()
    service.analyze(_damage("FB-1"))
    service.reject("FB-1", reviewer="ops-lead", reason="draft promised a call we cannot staff")
    nxt = service.analyze(_damage("FB-2", customer="Someone Else"))  # same lane, new consignee
    lines = _feedback_lines(nxt)
    assert len(lines) == 1
    assert "on this lane" in lines[0]
    assert "was reject" in lines[0]
    assert "draft promised a call we cannot staff" in lines[0]


def test_reject_reason_surfaces_on_the_next_same_consignee_case():
    service = _service()
    service.analyze(_damage("FB-3"))
    service.reject("FB-3", reviewer="ops-lead", reason="photos were not attached")
    nxt = service.analyze(
        _damage("FB-4", origin="Atlanta, GA", destination="Miami, FL")  # same consignee, new lane
    )
    lines = _feedback_lines(nxt)
    assert len(lines) == 1
    assert "for this consignee" in lines[0]
    assert "photos were not attached" in lines[0]


def test_approve_reason_is_stored_and_surfaces_too():
    service = _service()
    service.analyze(_damage("FB-5"))
    decided = service.approve("FB-5", approver="ops-lead", reason="verified with consignee by phone")
    assert decided.decision_reason == "verified with consignee by phone"
    nxt = service.analyze(_damage("FB-6"))
    lines = _feedback_lines(nxt)
    assert len(lines) == 1
    assert "was approve" in lines[0]
    assert "verified with consignee by phone" in lines[0]


def test_decision_reason_echoes_on_reject_and_defaults_to_none():
    service = _service()
    service.analyze(_damage("FB-7"))
    rejected = service.reject("FB-7", reviewer="ops-lead", reason="needs the claim form first")
    assert rejected.decision_reason == "needs the claim form first"
    service.analyze(_damage("FB-8"))
    approved = service.approve("FB-8", approver="ops-lead")
    assert approved.decision_reason is None


# --------------------------------------------------------------------------
# Bounds and silence
# --------------------------------------------------------------------------

def test_no_decisions_means_no_feedback_lines():
    service = _service()
    service.analyze(_damage("FB-9"))
    nxt = service.analyze(_damage("FB-10"))
    assert _feedback_lines(nxt) == []


def test_reasonless_decisions_teach_nothing():
    service = _service()
    service.analyze(_damage("FB-11"))
    service.reject("FB-11", reviewer="ops-lead")  # no reason given
    nxt = service.analyze(_damage("FB-12"))
    assert _feedback_lines(nxt) == []


def test_unrelated_decisions_do_not_surface():
    service = _service()
    service.analyze(_damage("FB-13"))
    service.reject("FB-13", reviewer="ops-lead", reason="wrong paperwork")
    unrelated = service.analyze(
        _damage("FB-14", customer="Other Co", origin="Tampa, FL", destination="Orlando, FL")
    )
    assert _feedback_lines(unrelated) == []


def test_feedback_is_bounded_to_the_last_three_matching_decisions():
    service = _service()
    for i in range(1, 5):
        service.analyze(_damage(f"FB-B{i}"))
        service.reject(f"FB-B{i}", reviewer="ops-lead", reason=f"reason number {i}")
    nxt = service.analyze(_damage("FB-B5"))
    lines = _feedback_lines(nxt)
    assert len(lines) == 3
    joined = " ".join(lines)
    assert "reason number 4" in joined  # newest first
    assert "reason number 2" in joined
    assert "reason number 1" not in joined  # the oldest is cut


def test_feedback_survives_across_service_instances_with_sqlite(tmp_path):
    db = tmp_path / "state.db"
    first = _service(SQLiteStore(db))
    first.analyze(_damage("FB-S1"))
    first.reject("FB-S1", reviewer="ops-lead", reason="carrier paperwork was stale")
    second = _service(SQLiteStore(db))
    nxt = second.analyze(_damage("FB-S2"))
    lines = _feedback_lines(nxt)
    assert len(lines) == 1
    assert "carrier paperwork was stale" in lines[0]
    # The stored record itself round-trips the decision + echo.
    record = second.get("FB-S1")
    assert record is not None
    assert record.approval_status == "rejected"
    assert record.decision_reason == "carrier paperwork was stale"


# --------------------------------------------------------------------------
# API surface: approve accepts a reason too; GET shows the decided result
# --------------------------------------------------------------------------

def test_api_approve_reason_echo_and_feedback_visibility():
    client = TestClient(app)
    shipment = {
        "shipment_id": "FB-API-1",
        "origin": "Raleigh, NC",
        "destination": "Norfolk, VA",
        "customer_name": "Feedback API Co",
        "scheduled_delivery": "2026-10-10T09:00:00",
        "estimated_delivery": "2026-10-11T09:00:00",
        "latest_event": "Delayed at hub",
        "documents": [],
    }
    assert client.post("/shipments/analyze", json=shipment).status_code == 200
    approved = client.post(
        "/shipments/FB-API-1/approve",
        json={"actor": "ops-lead", "reason": "customer confirmed the delay by phone"},
    )
    assert approved.status_code == 200
    assert approved.json()["decision_reason"] == "customer confirmed the delay by phone"
    fetched = client.get("/shipments/FB-API-1")
    assert fetched.status_code == 200
    assert fetched.json()["decision_reason"] == "customer confirmed the delay by phone"

    second = dict(shipment, shipment_id="FB-API-2")
    body = client.post("/shipments/analyze", json=second).json()
    evidence = body["diagnosis"]["evidence"]
    assert any(
        "reviewer feedback:" in line and "customer confirmed the delay by phone" in line
        for line in evidence
    )
