"""Diagnose node logic: a root-cause note over the computed evidence.

The diagnosis reasons over tool outputs only — computed delay hours,
document mismatches, extraction cross-check discrepancies, the
classification's own signals, and the retrieved policies — and cites
them. The evidence list is assembled deterministically in BOTH modes,
so the citations an approver sees are always the real inputs. In
provider mode the LLM composes the root-cause prose from those facts;
in the default mode a deterministic template composes it from the same
evidence structure. An unusable LLM reply falls back to the template.
"""

from __future__ import annotations

from .extractor import extraction_discrepancies
from .model_backends import DraftContext
from .schemas import Diagnosis, DocumentExtraction, ShipmentInput


def build_evidence(
    *,
    classification: dict,
    delay_hours: float | None,
    mismatches: list[dict],
    extractions: list[DocumentExtraction],
    policies: list[dict],
) -> list[str]:
    """The deterministic evidence list every diagnosis cites."""
    evidence: list[str] = []
    if delay_hours is not None:
        evidence.append(f"computed delay_hours={delay_hours} (tool: compute_delay_hours)")
    for m in mismatches:
        evidence.append(
            f"document mismatch on '{m['field']}': BOL={m.get('bol_value')} vs "
            f"invoice={m.get('invoice_value')} (tool: compare_documents)"
        )
    for line in extraction_discrepancies(extractions):
        evidence.append(f"extraction cross-check: {line}")
    if classification.get("signals"):
        evidence.append("classification signals: " + "; ".join(classification["signals"]))
    for p in policies:
        evidence.append(f"policy {p['policy_id']}: {p['title']}")
    return evidence


def _template_root_cause(
    *,
    shipment: ShipmentInput,
    classification: dict,
    delay_hours: float | None,
    mismatches: list[dict],
) -> str:
    exception = classification["exception_type"]
    if exception == "delay":
        cause = "The shipment is behind its scheduled delivery"
        if delay_hours is not None:
            cause += f" by a computed {delay_hours} hours"
        if shipment.latest_event:
            cause += f"; the carrier's latest report states: \"{shipment.latest_event}\""
        return cause + "."
    if exception == "damage":
        notes = shipment.condition_notes or shipment.latest_event or "condition record"
        return f"Physical damage was recorded against the shipment: \"{notes}\"."
    if exception == "document_mismatch":
        fields = ", ".join(m["field"] for m in mismatches) or "unspecified fields"
        return (
            f"The bill of lading and the invoice disagree on {fields}, so the "
            "shipment cannot clear final delivery until corrected documents are issued."
        )
    if exception == "missed_appointment":
        return (
            "The scheduled delivery appointment did not happen"
            + (f": \"{shipment.latest_event}\"." if shipment.latest_event else ".")
        )
    return "No exception evidence was found; the shipment is progressing within its schedule."


def build_diagnosis(
    *,
    shipment: ShipmentInput,
    classification: dict,
    delay_hours: float | None,
    mismatches: list[dict],
    extractions: list[DocumentExtraction],
    policies: list[dict],
    backend,
) -> Diagnosis:
    """Compose the diagnosis: LLM prose in provider mode, template otherwise."""
    evidence = build_evidence(
        classification=classification,
        delay_hours=delay_hours,
        mismatches=mismatches,
        extractions=extractions,
        policies=policies,
    )
    citations = [p["policy_id"] for p in policies]
    template = Diagnosis(
        root_cause=_template_root_cause(
            shipment=shipment,
            classification=classification,
            delay_hours=delay_hours,
            mismatches=mismatches,
        ),
        summary=(
            f"{classification['exception_type']} ({classification['severity']}) on "
            f"{shipment.origin} -> {shipment.destination}; {len(policies)} governing "
            f"polic{'y' if len(policies) == 1 else 'ies'} retrieved."
        ),
        evidence=evidence,
        citations=citations,
        source="template",
    )
    diagnose_fn = getattr(backend, "diagnose", None)
    if diagnose_fn is None:
        return template
    context = DraftContext(
        shipment_id=shipment.shipment_id,
        origin=shipment.origin,
        destination=shipment.destination,
        carrier=shipment.carrier,
        exception_type=classification["exception_type"],
        severity=classification["severity"],
        rationale=classification["rationale"],
        delay_hours=delay_hours,
        mismatches=mismatches,
        discrepancies=extraction_discrepancies(extractions),
        latest_event=shipment.latest_event,
        condition_notes=shipment.condition_notes,
        policy_details=policies,
    )
    try:
        llm = diagnose_fn(context)
    except Exception:  # the diagnosis never fails the run
        llm = None
    if not llm:
        return template
    return Diagnosis(
        root_cause=llm["root_cause"],
        summary=llm["summary"],
        evidence=evidence,
        citations=citations,
        source="llm",
    )
