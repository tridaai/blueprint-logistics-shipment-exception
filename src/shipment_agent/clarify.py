"""Clarification requests: the information-needed flow.

Some cases cannot be decided from the data on file: the classification
lands on ``none`` at low confidence AND something the review needs is
missing — the bill_of_lading/invoice pair (so the document-mismatch
check was skipped) or a critical document field the extraction could
not find in the document text. Rather than letting a human approver
discover that by reading the whole file, the pipeline composes a
clarification request to the carrier/ops contact listing *exactly*
what is missing.

The trigger and the missing-items list are deterministic code. Only
the wording is composed — by the LLM in provider mode, by a template
in the default mode, same shape either way. The request is *attached*
to the result (and the claim packet): the case still stops at the
human-approval gate with ``needs_information=True``. Nothing is sent —
sending it is the customer's integration step, like approving.
"""

from __future__ import annotations

from .model_backends import DraftContext
from .schemas import InformationRequest, ShipmentInput
from .tools import COMPARE_FIELDS

# Below this final-classification confidence, a "none" is not a finding
# — it is the absence of one, and combined with missing inputs it means
# the case is genuinely under-determined.
LOW_CONFIDENCE_CEILING = 0.6


def missing_items(
    shipment: ShipmentInput,
    document_check_warning: str | None,
    extractions: list[dict],
) -> list[str]:
    """Exactly what is missing, as plain list items. Deterministic.

    Two sources: the skipped document-pair check (which of BOL/invoice
    is absent) and extraction fields with status ``missing_in_text``
    over the critical compare fields (``tools.COMPARE_FIELDS``).
    """
    items: list[str] = []
    if document_check_warning:
        present = {d.doc_type for d in shipment.documents}
        if "bol" not in present:
            items.append("a bill of lading (BOL) for this shipment")
        if "invoice" not in present:
            items.append("the commercial invoice for this shipment")
    for extraction in extractions:
        for field in extraction.get("fields", []):
            if field.get("status") == "missing_in_text" and field.get("field") in COMPARE_FIELDS:
                items.append(
                    f"the '{field['field']}' value as it appears on document "
                    f"{extraction.get('document_id')} (the provided value was "
                    "not found in the document text)"
                )
    return items


def _template_message(shipment: ShipmentInput, items: list[str]) -> str:
    listing = "\n".join(f"- {item}" for item in items)
    return (
        f"Subject: Information needed — shipment {shipment.shipment_id} "
        f"({shipment.origin} -> {shipment.destination})\n\n"
        f"Hello {shipment.carrier} operations team,\n\n"
        "We are reviewing shipment "
        f"{shipment.shipment_id} and the data on file does not determine "
        "whether an exception occurred. To complete the review we still need:\n\n"
        f"{listing}\n\n"
        "Please send the missing item(s) to the operations contact on this "
        "case. The review stays open at our human-approval gate meanwhile; "
        "no customer update has been sent."
    )


def build_information_request(
    *,
    shipment: ShipmentInput,
    classification: dict,
    document_check_warning: str | None,
    extractions: list[dict],
    backend,
) -> InformationRequest | None:
    """Compose the clarification request, or ``None`` when not needed.

    Triggers only when the final classification is ``none`` below
    ``LOW_CONFIDENCE_CEILING`` *and* at least one concrete item is
    missing. Composition never fails the run: a provider failure falls
    back to the template wording with the same missing-items list.
    """
    if classification.get("exception_type") != "none":
        return None
    if float(classification.get("confidence", 1.0)) >= LOW_CONFIDENCE_CEILING:
        return None
    items = missing_items(shipment, document_check_warning, extractions)
    if not items:
        return None
    compose = getattr(backend, "compose_information_request", None)
    if compose is not None:
        context = DraftContext(
            shipment_id=shipment.shipment_id,
            origin=shipment.origin,
            destination=shipment.destination,
            carrier=shipment.carrier,
            customer_name=shipment.customer_name,
            missing_items=items,
        )
        try:
            message = compose(context)
        except Exception:  # composition never fails the run
            message = None
        if message:
            return InformationRequest(
                message=message, missing_items=items, source="llm"
            )
    return InformationRequest(
        message=_template_message(shipment, items),
        missing_items=items,
        source="template",
    )
