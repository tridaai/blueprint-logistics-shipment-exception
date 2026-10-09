"""Intake normalization tests.

Real intake (a TMS export, a carrier webhook) does not speak this
repo's exact dialect: doc_types arrive as bill_of_lading / BOL /
Invoice, and document field values arrive as numbers. Intake normalises
both instead of 422-ing the payload, and a skipped document-mismatch
check is always announced, never silent.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shipment_agent.api import app
from shipment_agent.graph import run_shipment
from shipment_agent.schemas import ShipmentInput

client = TestClient(app)


def _doc(doc_type: str, fields: dict | None = None, document_id: str = "D-1") -> dict:
    return {"doc_type": doc_type, "document_id": document_id, "fields": fields or {}}


def _shipment(documents: list[dict], **overrides) -> dict:
    payload = {
        "shipment_id": "INTAKE-1",
        "origin": "Memphis, TN",
        "destination": "Charlotte, NC",
        "documents": documents,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "provided, canonical",
    [
        ("bol", "bol"),
        ("BOL", "bol"),
        ("bill_of_lading", "bol"),
        ("Bill of Lading", "bol"),
        ("bill-of-lading", "bol"),
        ("invoice", "invoice"),
        ("Invoice", "invoice"),
        ("commercial_invoice", "invoice"),
        ("tracking", "tracking"),
        ("delivery_note", "delivery_note"),
        ("POD", "delivery_note"),
        ("proof of delivery", "delivery_note"),
        ("other", "other"),
    ],
)
def test_doc_type_aliases_normalize(provided: str, canonical: str):
    shipment = ShipmentInput.model_validate(_shipment([_doc(provided)]))
    document = shipment.documents[0]
    assert document.doc_type == canonical
    assert document.doc_type_flagged is False
    assert document.doc_type_provided == provided


def test_unknown_doc_type_is_kept_but_flagged():
    shipment = ShipmentInput.model_validate(_shipment([_doc("Customs Form")]))
    document = shipment.documents[0]
    assert document.doc_type == "customs_form"
    assert document.doc_type_flagged is True
    assert document.doc_type_provided == "Customs Form"


def test_numeric_field_values_coerce_to_strings():
    shipment = ShipmentInput.model_validate(
        _shipment([_doc("bol", {"quantity_units": 120, "weight_kg": 840.5, "sku": "SKU-1"})])
    )
    assert shipment.documents[0].fields == {
        "quantity_units": "120",
        "weight_kg": "840.5",
        "sku": "SKU-1",
    }


def test_integral_float_field_value_coerces_without_decimal():
    shipment = ShipmentInput.model_validate(
        _shipment([_doc("bol", {"weight_kg": 840.0})])
    )
    assert shipment.documents[0].fields["weight_kg"] == "840"


def test_aliased_pair_still_produces_mismatches():
    """bill_of_lading + Invoice must be found as the comparison pair."""
    shipment = _shipment(
        [
            _doc("bill_of_lading", {"quantity_units": 120, "weight_kg": 840}, "BOL-9"),
            _doc("Invoice", {"quantity_units": 100, "weight_kg": 840}, "INV-9"),
        ]
    )
    result = run_shipment(ShipmentInput.model_validate(shipment))
    assert [m.field for m in result.document_mismatches] == ["quantity_units"]
    assert result.document_mismatches[0].bol_value == "120"
    assert result.document_mismatches[0].invoice_value == "100"
    assert result.document_check_warning is None


def test_missing_pair_warns_in_result_and_trace():
    shipment = _shipment([_doc("tracking", {}, "TRK-1")], latest_event="In transit")
    result = run_shipment(ShipmentInput.model_validate(shipment))
    assert result.document_mismatches == []
    assert result.document_check_warning is not None
    assert result.document_check_warning.startswith(
        "no bill_of_lading/invoice pair found — document mismatch check skipped"
    )
    ingest_step = next(s for s in result.trace if s.name == "ingest")
    assert any("no bill_of_lading/invoice pair found" in d for d in ingest_step.details)


def test_numeric_fields_do_not_422_via_api():
    shipment = _shipment(
        [
            _doc("bill_of_lading", {"quantity_units": 120}, "BOL-9"),
            _doc("invoice", {"quantity_units": 120}, "INV-9"),
        ],
        shipment_id="INTAKE-API-1",
        latest_event="Delayed at hub",
    )
    response = client.post("/shipments/analyze", json=shipment)
    assert response.status_code == 200
    assert response.json()["document_check_warning"] is None
