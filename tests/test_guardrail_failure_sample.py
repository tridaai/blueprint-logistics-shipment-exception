"""SYN-1013 — the bundled guardrail-failure sample.

Its carrier condition note carries promise language ("we guarantee a full
refund"). The drafting template quotes the source record, so the draft
inherits the banned phrases and the ``no_prohibited_promises`` guardrail
blocks it: validation fails, approval is refused, in every mode. This is
the demo's trust moment — a guardrail visibly catching a bad draft.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shipment_agent.api import app
from shipment_agent.graph import run_shipment
from shipment_agent.samples import load_sample_shipments
from shipment_agent.schemas import ShipmentInput
from shipment_agent.service import ShipmentService

client = TestClient(app)


def _sample(shipment_id: str) -> dict:
    return next(s for s in load_sample_shipments() if s["shipment_id"] == shipment_id)


def test_syn1013_is_in_the_sample_list():
    ids = {s["shipment_id"] for s in load_sample_shipments()}
    assert "SYN-1013" in ids
    api_ids = {s["shipment_id"] for s in client.get("/samples").json()}
    assert "SYN-1013" in api_ids  # the demo console lists it too


def test_syn1013_draft_fails_no_prohibited_promises():
    result = run_shipment(ShipmentInput.model_validate(_sample("SYN-1013")))
    assert result.classification.exception_type.value == "delay"
    assert result.validation.passed is False
    checks = {c.name: c for c in result.validation.checks}
    assert checks["no_prohibited_promises"].passed is False
    assert "full refund" in checks["no_prohibited_promises"].detail
    # Only the promise check fails — the draft is otherwise well-formed.
    assert checks["references_shipment_id"].passed is True
    assert checks["policy_citations_present"].passed is True
    assert any("prohibited promise" in e for e in result.validation.errors)


def test_syn1013_stops_at_the_gate_with_nothing_sent():
    result = run_shipment(ShipmentInput.model_validate(_sample("SYN-1013")))
    assert result.approval_status == "awaiting_approval"
    assert result.external_action_taken is False
    validate_step = result.trace[4]
    assert validate_step.status == "failed"
    assert any("no_prohibited_promises" in d for d in validate_step.details)


def test_syn1013_cannot_be_approved_via_service():
    service = ShipmentService()
    service.analyze(_sample("SYN-1013"))
    with pytest.raises(ValueError, match="failed guardrail validation"):
        service.approve("SYN-1013", approver="ops-lead")


def test_syn1013_approve_returns_422_via_api():
    shipment = _sample("SYN-1013")
    analyzed = client.post("/shipments/analyze", json=shipment)
    assert analyzed.status_code == 200
    assert analyzed.json()["validation"]["passed"] is False
    response = client.post("/shipments/SYN-1013/approve", json={"approver": "ops-lead"})
    assert response.status_code == 422


def test_all_other_samples_pass_guardrails():
    """The mock's verbatim quoting of source notes must not break any
    other bundled sample — SYN-1013 is the only designed failure."""
    for sample in load_sample_shipments():
        if sample["shipment_id"] == "SYN-1013":
            continue
        result = run_shipment(ShipmentInput.model_validate(sample))
        assert result.validation.passed, sample["shipment_id"]
