"""Deterministic tools used by the agent.

Tools are plain, testable functions. The agent never lets a model guess at
facts a function can compute exactly (delay hours, document mismatches).
"""

from __future__ import annotations

from datetime import datetime

from .schemas import DocumentInput, DocumentMismatch

# Fields that must agree between the bill of lading and the invoice.
_COMPARE_FIELDS = ("quantity_units", "weight_kg", "consignee", "sku")


def compute_delay_hours(
    scheduled: datetime | None, estimated: datetime | None
) -> float | None:
    """Hours between scheduled and estimated delivery (negative = early)."""
    if scheduled is None or estimated is None:
        return None
    return round((estimated - scheduled).total_seconds() / 3600.0, 2)


def _find_doc(documents: list[DocumentInput], doc_type: str) -> DocumentInput | None:
    return next((d for d in documents if d.doc_type.lower() == doc_type), None)


def compare_documents(documents: list[DocumentInput]) -> list[DocumentMismatch]:
    """Compare BOL vs invoice structured fields; return every disagreement."""
    bol = _find_doc(documents, "bol")
    invoice = _find_doc(documents, "invoice")
    if bol is None or invoice is None:
        return []
    mismatches: list[DocumentMismatch] = []
    for field in _COMPARE_FIELDS:
        bol_value = bol.fields.get(field)
        inv_value = invoice.fields.get(field)
        if bol_value is not None and inv_value is not None and bol_value.strip() != inv_value.strip():
            mismatches.append(
                DocumentMismatch(field=field, bol_value=bol_value, invoice_value=inv_value)
            )
    return mismatches


def combined_event_text(latest_event: str, condition_notes: str, status: str) -> str:
    """All free-text signal about the shipment, lower-cased for rule matching."""
    return f"{status} {latest_event} {condition_notes}".lower()
