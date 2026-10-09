"""End-to-end graph tests — fully offline, mock backend, synthetic data."""

from shipment_agent.graph import run_shipment
from shipment_agent.schemas import ExceptionType, ShipmentInput

DELAY_SHIPMENT = {
    "shipment_id": "INT-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


def test_end_to_end_delay_stops_at_approval_gate():
    result = run_shipment(ShipmentInput.model_validate(DELAY_SHIPMENT))
    assert result.classification.exception_type == ExceptionType.DELAY
    assert result.delay_hours == 36.0
    assert result.policies, "expected at least one retrieved policy"
    assert "INT-1" in result.draft.body
    assert result.validation.passed
    assert result.approval_status == "awaiting_approval"
    # The core safety property of the blueprint:
    assert result.external_action_taken is False


def test_end_to_end_accepts_plain_dict():
    result = run_shipment(DELAY_SHIPMENT)
    assert result.shipment_id == "INT-1"


def test_claim_packet_is_draft_not_filed():
    result = run_shipment(DELAY_SHIPMENT)
    assert result.draft.claim_packet["status"] == "draft — not filed"
