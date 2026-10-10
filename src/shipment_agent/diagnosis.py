"""Diagnose node logic: a root-cause note over the computed evidence.

The diagnosis reasons over tool outputs only — computed delay hours,
document mismatches, extraction cross-check discrepancies, the
classification's own signals, and the retrieved policies — and cites
them. The evidence list is assembled deterministically in BOTH modes,
so the citations an approver sees are always the real inputs. In
provider mode the LLM composes the root-cause prose from those facts —
agenticly, through a bounded tool loop when the graph supplies a
toolbox (``tools_agent.py``); in the default mode a deterministic
template composes it from the same evidence structure. An unusable
LLM reply falls back to the template.
"""

from __future__ import annotations

from .extractor import extraction_discrepancies
from .model_backends import DraftContext
from .schemas import Diagnosis, DocumentExtraction, ShipmentInput
from .screening import sanitized_event_notes
from .store import format_type_counts
from .tools_agent import TOOL_SPECS, DiagnosisToolBox, diagnosis_max_tool_calls


def memory_evidence_lines(history: dict | None) -> list[str]:
    """History as evidence lines, in the diagnosis' citation style.

    Only genuine exceptions count (a prior "none" is not an exception),
    and the lines name the counts plainly so an approver can weigh them.
    Empty history produces no lines at all — absence is not evidence.
    """
    if not history:
        return []
    lines: list[str] = []
    if history.get("consignee_count"):
        recent = ", ".join(history.get("consignee_recent_types", [])) or "none recorded"
        lines.append(
            f"memory: {history['consignee_count']} prior exception(s) for this "
            f"consignee in the stored history (consignee: {history.get('consignee', '')}; "
            f"most recent: {recent})"
        )
    if history.get("lane_count"):
        recent = ", ".join(history.get("lane_recent_types", [])) or "none recorded"
        lines.append(
            f"memory: {history['lane_count']} prior exception(s) on this lane "
            f"({history.get('lane', '')}) in the stored history (most recent: {recent})"
        )
    if history.get("carrier_exception_count"):
        breakdown = format_type_counts(history.get("carrier_type_counts", {}))
        lines.append(
            f"carrier history: {history['carrier_count']} prior shipment(s) with "
            f"this carrier ({history.get('carrier', '')}) in the stored history "
            f"(exceptions: {breakdown})"
        )
    return lines


def build_evidence(
    *,
    classification: dict,
    delay_hours: float | None,
    mismatches: list[dict],
    extractions: list[DocumentExtraction],
    policies: list[dict],
    history: dict | None = None,
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
    evidence.extend(memory_evidence_lines(history))
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
    history: dict | None = None,
    toolbox: DiagnosisToolBox | None = None,
) -> Diagnosis:
    """Compose the diagnosis: LLM prose in provider mode, template otherwise.

    In provider mode with a ``toolbox`` (the graph passes one), the LLM
    diagnosis is agentic: a bounded tool loop (``tools_agent.py``) lets
    the model pull policy search results, history, and computed facts
    before composing. Without a toolbox the single-call diagnosis runs,
    exactly as before. Either path degrades to the template, with the
    reason recorded in ``note``.
    """
    # The diagnosis quotes the carrier's text (template path) and feeds
    # it to the model (provider path): both see the screened version —
    # sentences the injection screen flagged never reach either.
    event_text, notes_text = sanitized_event_notes(shipment)
    if (event_text, notes_text) != (shipment.latest_event, shipment.condition_notes):
        shipment = shipment.model_copy(
            update={"latest_event": event_text, "condition_notes": notes_text}
        )
    evidence = build_evidence(
        classification=classification,
        delay_hours=delay_hours,
        mismatches=mismatches,
        extractions=extractions,
        policies=policies,
        history=history,
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
    tools_fn = getattr(backend, "diagnose_with_tools", None)
    if tools_fn is not None and toolbox is not None:
        try:
            llm = tools_fn(
                context, TOOL_SPECS, toolbox.dispatch, diagnosis_max_tool_calls()
            )
        except Exception as exc:  # the diagnosis never fails the run
            llm = None
            template.note = (
                f"LLM agentic diagnosis failed ({exc}) — template diagnosis used"
            )
        if llm:
            return Diagnosis(
                root_cause=llm["root_cause"],
                summary=llm["summary"],
                evidence=evidence,
                citations=citations,
                source="llm",
                tool_calls=llm.get("tool_calls", []),
            )
        if not template.note:
            template.note = (
                "LLM agentic diagnosis reply was unusable — template diagnosis used"
            )
        return template
    try:
        llm = diagnose_fn(context)
    except Exception as exc:  # the diagnosis never fails the run
        llm = None
        template.note = f"LLM diagnosis failed ({exc}) — template diagnosis used"
    if not llm:
        if not template.note:
            template.note = "LLM diagnosis reply was unusable — template diagnosis used"
        return template
    return Diagnosis(
        root_cause=llm["root_cause"],
        summary=llm["summary"],
        evidence=evidence,
        citations=citations,
        source="llm",
    )
