"""Negation and robustness cases — the inputs real-world testers try first.

Every case here started as something a person would plausibly type into
the demo UI. "No damage reported" must NOT classify as damage.
"""

from datetime import datetime, timedelta

from shipment_agent.classifier import classify_shipment
from shipment_agent.schemas import ExceptionType, ShipmentInput

BASE = datetime(2026, 10, 10, 9, 0, 0)


def make(**overrides) -> ShipmentInput:
    data = dict(shipment_id="NEG-1", origin="A", destination="B", latest_event="In transit")
    data.update(overrides)
    return ShipmentInput(**data)


def classify(shipment: ShipmentInput):
    return classify_shipment(shipment, []).exception_type


def test_no_damage_reported_is_not_damage():
    s = make(condition_notes="No damage reported at inspection")
    assert classify(s) == ExceptionType.NONE


def test_not_damaged_is_not_damage():
    s = make(condition_notes="Cargo inspected — not damaged, seals intact")
    assert classify(s) == ExceptionType.NONE


def test_undamaged_is_not_damage():
    s = make(latest_event="Delivered undamaged and on time")
    assert classify(s) == ExceptionType.NONE


def test_damage_colon_none_is_not_damage():
    s = make(condition_notes="Delivery inspection — damage: none")
    assert classify(s) == ExceptionType.NONE


def test_no_leak_found_is_not_damage():
    s = make(latest_event="Inspection complete: no leak found, no damage")
    assert classify(s) == ExceptionType.NONE


def test_noted_does_not_read_as_negation():
    # "noted" contains "no" as a substring — it is not a negation cue.
    s = make(condition_notes="Crushed cartons noted at terminal inspection")
    assert classify(s) == ExceptionType.DAMAGE


def test_negated_damage_but_real_delay_still_delay():
    s = make(
        condition_notes="No damage reported",
        scheduled_delivery=BASE,
        estimated_delivery=BASE + timedelta(hours=30),
    )
    assert classify(s) == ExceptionType.DELAY


def test_no_discrepancy_is_not_mismatch():
    s = make(latest_event="Documents checked, no discrepancy found")
    assert classify(s) == ExceptionType.NONE


def test_no_delay_is_not_delay():
    s = make(latest_event="Running to plan, no delay expected")
    assert classify(s) == ExceptionType.NONE


def test_recovered_delay_is_none_when_eta_back_on_schedule():
    s = make(
        latest_event="Earlier weather delay cleared at hub — back on schedule",
        scheduled_delivery=BASE,
        estimated_delivery=BASE + timedelta(minutes=30),
    )
    assert classify(s) == ExceptionType.NONE


def test_recovery_phrase_does_not_cancel_computed_delay():
    s = make(
        latest_event="Crew says back on schedule soon",
        scheduled_delivery=BASE,
        estimated_delivery=BASE + timedelta(hours=30),
    )
    assert classify(s) == ExceptionType.DELAY


def test_partial_damage_is_damage_at_medium_severity():
    from shipment_agent.schemas import Severity

    s = make(condition_notes="One carton dented, minor cosmetic damage to packaging")
    result = classify_shipment(s, [])
    assert result.exception_type == ExceptionType.DAMAGE
    assert result.severity == Severity.MEDIUM


def test_new_signal_phrases():
    assert classify(make(latest_event="Trailer punctured a tyre, freight soaked")) == ExceptionType.DAMAGE
    assert classify(make(latest_event="Receiver closed — delivery refused")) == ExceptionType.MISSED_APPOINTMENT
    assert classify(make(latest_event="Running behind schedule due to congestion")) == ExceptionType.DELAY
    assert classify(make(latest_event="Qty mismatch between BOL and invoice")) == ExceptionType.DOCUMENT_MISMATCH
