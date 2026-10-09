from datetime import datetime, timedelta

from shipment_agent.schemas import DocumentInput
from shipment_agent.tools import compare_documents, compute_delay_hours

BASE = datetime(2026, 10, 10, 9, 0, 0)


def test_compute_delay_hours():
    assert compute_delay_hours(BASE, BASE + timedelta(hours=5, minutes=30)) == 5.5
    assert compute_delay_hours(BASE, None) is None
    assert compute_delay_hours(None, BASE) is None


def test_compare_documents_detects_and_ignores():
    docs = [
        DocumentInput(
            doc_type="bol", document_id="B",
            fields={"quantity_units": "100", "weight_kg": "500", "consignee": "Acme"},
        ),
        DocumentInput(
            doc_type="invoice", document_id="I",
            fields={"quantity_units": "100", "weight_kg": "450", "consignee": "Acme"},
        ),
    ]
    mismatches = compare_documents(docs)
    assert len(mismatches) == 1
    assert mismatches[0].field == "weight_kg"


def test_compare_documents_missing_pair():
    docs = [DocumentInput(doc_type="bol", document_id="B", fields={"quantity_units": "1"})]
    assert compare_documents(docs) == []
