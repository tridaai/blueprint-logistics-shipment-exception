"""Document extraction node logic.

In provider mode the LLM reads each document's text and extracts the
typed fields (``tools.COMPARE_FIELDS``) with a per-field confidence —
structured output, via ``ModelBackend.extract_document_fields``.
Deterministic code then cross-checks every extracted value against the
field values the source system provided:

- ``match`` — extracted equals provided.
- ``mismatch`` — both present, different values (a data-quality signal
  the approver sees; the BOL-vs-invoice diff in ``tools.py`` remains
  the authority for document-vs-document disagreements).
- ``missing_in_text`` — provided, but the extractor could not find it
  in the document text.
- ``extracted_only`` — found in the text, absent from the provided fields.

In the default (mock) mode no extraction runs: fields pass through as
``source_provided`` with no confidence claimed. Extraction never fails
a run — a provider error or unusable reply falls back to the provided
fields for that document.
"""

from __future__ import annotations

from .schemas import DocumentExtraction, DocumentInput, FieldExtraction, ShipmentInput
from .tools import COMPARE_FIELDS


def _provided_extraction(document: DocumentInput) -> DocumentExtraction:
    return DocumentExtraction(
        document_id=document.document_id,
        doc_type=document.doc_type,
        source="provided",
        fields=[
            FieldExtraction(
                field=field,
                provided_value=document.fields.get(field),
                extracted_value=document.fields.get(field),
                confidence=None,
                status="source_provided",
            )
            for field in COMPARE_FIELDS
            if document.fields.get(field) is not None
        ],
    )


def _cross_check(document: DocumentInput, extracted: dict[str, dict]) -> DocumentExtraction:
    fields: list[FieldExtraction] = []
    for field in COMPARE_FIELDS:
        provided = document.fields.get(field)
        entry = extracted.get(field) or {"value": None, "confidence": 0.0}
        value = entry.get("value")
        confidence = entry.get("confidence")
        if provided is None and value is None:
            continue
        if value is None:
            status, confidence = "missing_in_text", None
        elif provided is None:
            status = "extracted_only"
        elif str(provided).strip() == str(value).strip():
            status = "match"
        else:
            status = "mismatch"
        fields.append(
            FieldExtraction(
                field=field,
                provided_value=provided,
                extracted_value=value,
                confidence=confidence,
                status=status,
            )
        )
    return DocumentExtraction(
        document_id=document.document_id,
        doc_type=document.doc_type,
        source="llm",
        fields=fields,
    )


def extract_documents(shipment: ShipmentInput, backend) -> list[DocumentExtraction]:
    """Extract + cross-check every document on the shipment.

    ``backend`` is the active model backend; only real LLM backends
    expose ``extract_document_fields``, so the default mode is a pure
    pass-through by construction.
    """
    extract = getattr(backend, "extract_document_fields", None)
    results: list[DocumentExtraction] = []
    for document in shipment.documents:
        if extract is None or not document.raw_text.strip():
            results.append(_provided_extraction(document))
            continue
        try:
            extracted = extract(
                document.doc_type,
                document.document_id,
                document.raw_text,
                list(COMPARE_FIELDS),
            )
        except Exception as exc:  # extraction is evidence, never a run failure
            fallback = _provided_extraction(document)
            fallback.note = (
                f"LLM extraction failed ({exc}) — provided fields used instead"
            )
            results.append(fallback)
            continue
        if extracted is None:
            fallback = _provided_extraction(document)
            fallback.note = "LLM extraction reply was unusable — provided fields used instead"
            results.append(fallback)
        else:
            results.append(_cross_check(document, extracted))
    return results


def _discrepancy_lines(document_id: str, fields) -> list[str]:
    lines: list[str] = []
    for field in fields:
        if field.status == "mismatch":
            lines.append(
                f"{document_id} field '{field.field}': provided "
                f"{field.provided_value!r} but document text says {field.extracted_value!r}"
            )
        elif field.status == "missing_in_text":
            lines.append(
                f"{document_id} field '{field.field}': provided value "
                f"{field.provided_value!r} not found in the document text"
            )
    return lines


def extraction_discrepancies(extractions: list[DocumentExtraction]) -> list[str]:
    """Human-readable discrepancy lines for the trace and the diagnosis."""
    lines: list[str] = []
    for extraction in extractions:
        lines += _discrepancy_lines(extraction.document_id, extraction.fields)
    return lines


def discrepancies_from_dicts(extractions: list[dict]) -> list[str]:
    """Same lines, from serialised state (graph trace building)."""
    from types import SimpleNamespace

    lines: list[str] = []
    for extraction in extractions:
        fields = [SimpleNamespace(**f) for f in extraction["fields"]]
        lines += _discrepancy_lines(extraction["document_id"], fields)
    return lines
