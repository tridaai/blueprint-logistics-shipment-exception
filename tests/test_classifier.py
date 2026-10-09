from datetime import datetime, timedelta

from shipment_agent.classifier import classify_shipment
from shipment_agent.schemas import DocumentInput, ExceptionType, Severity, ShipmentInput
from shipment_agent.tools import compare_documents

BASE = datetime(2026, 10, 10, 9, 0, 0)


def make_shipment(**overrides) -> ShipmentInput:
    data = dict(shipment_id="T-1", origin="A", destination="B", latest_event="In transit")
    data.update(overrides)
    return ShipmentInput(**data)


def test_delay_from_hours():
    s = make_shipment(
        scheduled_delivery=BASE, estimated_delivery=BASE + timedelta(hours=30)
    )
    result = classify_shipment(s, [])
    assert result.exception_type == ExceptionType.DELAY
    assert result.severity == Severity.HIGH


def test_damage_beats_delay():
    s = make_shipment(
        condition_notes="Cartons crushed",
        scheduled_delivery=BASE,
        estimated_delivery=BASE + timedelta(hours=30),
    )
    result = classify_shipment(s, [])
    assert result.exception_type == ExceptionType.DAMAGE
    assert result.severity == Severity.HIGH


def test_missed_appointment_beats_generic_late():
    s = make_shipment(latest_event="Driver late: missed delivery appointment at receiver")
    result = classify_shipment(s, [])
    assert result.exception_type == ExceptionType.MISSED_APPOINTMENT


def test_document_mismatch_from_fields():
    docs = [
        DocumentInput(doc_type="bol", document_id="B", fields={"quantity_units": "100"}),
        DocumentInput(doc_type="invoice", document_id="I", fields={"quantity_units": "90"}),
    ]
    s = make_shipment(documents=docs)
    result = classify_shipment(s, compare_documents(docs))
    assert result.exception_type == ExceptionType.DOCUMENT_MISMATCH
    assert result.confidence >= 0.95


def test_no_exception():
    s = make_shipment(
        scheduled_delivery=BASE, estimated_delivery=BASE + timedelta(hours=1)
    )
    result = classify_shipment(s, [])
    assert result.exception_type == ExceptionType.NONE
