"""Autonomy recommendation tests.

Every result carries a deterministic routing recommendation: eligible
for auto-approval only when the exception is none/low severity, the
guardrails passed, the cross-check did not disagree, and no repair was
needed. It is printed on the result, the trace, and the claim packet —
and it NEVER acts: the approval gate is unchanged.
"""

from __future__ import annotations

from shipment_agent.autonomy import compute_autonomy
from shipment_agent.graph import run_shipment
from shipment_agent.samples import load_sample_shipments
from shipment_agent.schemas import ShipmentInput

NONE_SHIPMENT = {
    "shipment_id": "AUT-0",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-10T09:00:00",
    "latest_event": "In transit, on schedule",
    "documents": [],
}


def _sample(shipment_id: str) -> dict:
    return next(s for s in load_sample_shipments() if s["shipment_id"] == shipment_id)


# --------------------------------------------------------------------------
# Graph level
# --------------------------------------------------------------------------

def test_none_exception_is_eligible_and_still_waits_for_a_human():
    result = run_shipment(ShipmentInput.model_validate(NONE_SHIPMENT))
    assert result.classification.exception_type.value == "none"
    assert result.autonomy is not None
    assert result.autonomy.eligible_for_auto_approval is True
    # The recommendation never acts: the gate is exactly where it was.
    assert result.approval_status == "awaiting_approval"
    assert result.external_action_taken is False
    packet = result.draft.claim_packet["autonomy_recommendation"]
    assert packet["eligible_for_auto_approval"] is True
    gate = next(s for s in result.trace if s.name == "human_approval")
    assert any("autonomy recommendation: eligible" in d for d in gate.details)


def test_high_severity_is_not_eligible():
    result = run_shipment(ShipmentInput.model_validate(_sample("SYN-1001")))
    assert result.autonomy is not None
    assert result.autonomy.eligible_for_auto_approval is False
    assert any("severity" in r for r in result.autonomy.reasons)


# --------------------------------------------------------------------------
# Unit level — each disqualifier on its own
# --------------------------------------------------------------------------

BASE = {
    "classification": {"exception_type": "none", "severity": "low"},
    "validation": {"passed": True},
    "cross_check": None,
    "repair_attempted": False,
}


def test_eligible_baseline():
    rec = compute_autonomy(**BASE)
    assert rec.eligible_for_auto_approval is True
    assert len(rec.reasons) == 5


def test_disqualifier_severity():
    rec = compute_autonomy(
        **{**BASE, "classification": {"exception_type": "delay", "severity": "high"}}
    )
    assert rec.eligible_for_auto_approval is False
    assert any("above the auto-approval band" in r for r in rec.reasons)


def test_low_severity_exception_is_in_band():
    rec = compute_autonomy(
        **{**BASE, "classification": {"exception_type": "delay", "severity": "low"}}
    )
    assert rec.eligible_for_auto_approval is True


def test_disqualifier_guardrail_failure():
    rec = compute_autonomy(**{**BASE, "validation": {"passed": False}})
    assert rec.eligible_for_auto_approval is False
    assert any("guardrails did not pass" in r for r in rec.reasons)


def test_disqualifier_cross_check_disagreement():
    for resolution in ("rules_authoritative", "llm_adopted"):
        rec = compute_autonomy(
            **{**BASE, "cross_check": {"resolution": resolution}}
        )
        assert rec.eligible_for_auto_approval is False, resolution
        assert any("disagreed" in r for r in rec.reasons)


def test_cross_check_agreement_keeps_eligibility():
    rec = compute_autonomy(**{**BASE, "cross_check": {"resolution": "agree"}})
    assert rec.eligible_for_auto_approval is True
    rec = compute_autonomy(**{**BASE, "cross_check": {"resolution": "rules_only"}})
    assert rec.eligible_for_auto_approval is True


def test_disqualifier_repair():
    rec = compute_autonomy(**{**BASE, "repair_attempted": True})
    assert rec.eligible_for_auto_approval is False
    assert any("repair" in r for r in rec.reasons)
