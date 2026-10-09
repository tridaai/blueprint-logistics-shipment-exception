"""The agent graph (LangGraph).

Pipeline — every step is a named, inspectable node:

    extract ──► ingest ──► classify ──► retrieve ──► draft ──► validate
        ──► human_approval ──► END

The graph deliberately has NO node that sends a message, files a claim, or
touches an external system. It ends at the human-approval gate. Acting on an
approved draft is a separate, explicit step (see ``service.py``), which in
this prototype still performs no external action.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .classifier import classify_shipment
from .crosscheck import resolve_classification
from .extractor import discrepancies_from_dicts, extract_documents
from .guardrails import validate_draft
from .model_backends import DraftContext, ModelBackend, MockModelBackend
from .retriever import KeywordRetriever, Retriever
from .schemas import AgentResult, DraftOutput, ExceptionType, ShipmentInput, TraceStep
from .tools import compare_documents, compute_delay_hours

# Sentinel distinguishing "no LLM backend configured" (default mode — no
# cross-check is recorded at all) from "LLM configured but its reply was
# unusable" (cross-check records rules_only). See crosscheck.py for the
# resolution policy itself.
_NO_LLM_BACKEND = object()


def _build_trace(shipment: ShipmentInput, final: dict) -> list[TraceStep]:
    """Assemble the inspectable per-step trace from the final graph state."""
    classification = final["classification"]
    rule_classification = final.get("rule_classification") or classification
    cross_check = final.get("cross_check")
    classify_details = [f"evidence: {s}" for s in classification.get("signals", [])] or ["evidence: (no signals)"]
    classify_details.append(
        f"rules: {rule_classification['exception_type']} · severity "
        f"{rule_classification['severity']} · confidence {rule_classification['confidence']}"
    )
    if cross_check:
        if cross_check.get("llm_exception_type"):
            classify_details.append(
                f"llm cross-check ({cross_check['llm_backend']}): "
                f"{cross_check['llm_exception_type']} · severity {cross_check['llm_severity']} · "
                f"confidence {cross_check['llm_confidence']}"
            )
        else:
            classify_details.append(
                "llm cross-check: no usable LLM classification (provider error or "
                "unusable reply) — rules only"
            )
        resolution_lines = {
            "agree": "cross-check: AGREE — both paths classified this shipment the same way",
            "rules_authoritative": (
                "cross-check: DISAGREEMENT — the rule result stays authoritative; "
                "the disagreement is recorded for the approver"
            ),
            "llm_adopted": (
                "cross-check: DISAGREEMENT — LLM classification ADOPTED (rules were "
                "none/low-confidence, LLM highly confident) and flagged for the approver"
            ),
            "rules_only": "cross-check: rules only — no LLM classification available",
        }
        classify_details.append(resolution_lines[cross_check["resolution"]])
    mismatches = final.get("document_mismatches", [])
    delay = final.get("delay_hours")
    policies = final.get("policies", [])
    validation = final["validation"]

    extract_details: list[str] = []
    extractions = final.get("extractions", [])
    for extraction in extractions:
        if extraction["source"] == "llm":
            extract_details.append(
                f"tool call: extract_document_fields({extraction['document_id']}) "
                f"-> {len(extraction['fields'])} field(s) with confidences"
            )
            for f in extraction["fields"]:
                confidence = f" · confidence {f['confidence']}" if f["confidence"] is not None else ""
                extract_details.append(
                    f"{f['field']}: provided {f['provided_value']!r} vs extracted "
                    f"{f['extracted_value']!r}{confidence} — {f['status']}"
                )
        else:
            extract_details.append(
                f"{extraction['document_id']}: fields source-provided "
                "(LLM extraction runs only with a provider backend configured)"
            )
    extract_details += [
        f"discrepancy — {d}" for d in discrepancies_from_dicts(extractions)
    ]
    if not extract_details:
        extract_details.append("no documents on this shipment — nothing to extract")

    ingest_details = [
        f"tool call: compute_delay_hours(scheduled_delivery, estimated_delivery) -> {delay}",
        f"tool call: compare_documents(bol, invoice) -> {len(mismatches)} mismatch(es)",
        f"documents compared: {len(shipment.documents)}",
    ]
    for m in mismatches:
        ingest_details.append(
            f"mismatch — {m['field']}: BOL={m.get('bol_value')} vs invoice={m.get('invoice_value')}"
        )
    if not mismatches:
        ingest_details.append("no document mismatches found")

    return [
        TraceStep(
            name="extract",
            title="Extract document fields",
            summary=(
                "Typed fields extracted from document text with per-field confidence "
                "and cross-checked against the provided values by code."
                if any(e["source"] == "llm" for e in extractions)
                else "Document fields taken as source-provided (LLM extraction runs in provider mode)."
            ),
            details=extract_details,
        ),
        TraceStep(
            name="ingest",
            title="Ingest & compute facts",
            summary="Validated shipment; delay hours and document diffs computed by code, never by a model.",
            details=ingest_details,
        ),
        TraceStep(
            name="classify",
            title="Classify exception",
            summary=(
                f"{classification['exception_type']} · severity {classification['severity']} · "
                f"confidence {classification['confidence']} — {classification['rationale']}"
            ),
            details=classify_details,
        ),
        TraceStep(
            name="retrieve",
            title="Retrieve policy context",
            summary=f"{len(policies)} policy snippet(s) retrieved from the synthetic SOP corpus.",
            details=[
                f"[{p['policy_id']}] {p['title']} (score {p['score']}): {p['snippet']}"
                for p in policies
            ],
        ),
        TraceStep(
            name="draft",
            title="Draft update & claim packet",
            summary=f"Customer update drafted with {len(final['draft']['citations'])} policy citation(s); claim packet assembled as draft — not filed.",
            details=[f"subject: {final['draft']['subject']}"],
        ),
        TraceStep(
            name="validate",
            title="Guardrail validation",
            status="passed" if validation["passed"] else "failed",
            summary="Draft checked against guardrails implemented as code.",
            details=[
                f"{'PASS' if c['passed'] else 'FAIL'} {c['name']} — {c['detail']}"
                for c in validation.get("checks", [])
            ]
            + [f"warning: {w}" for w in validation.get("warnings", [])],
        ),
        TraceStep(
            name="human_approval",
            title="Human approval gate",
            status="awaiting",
            summary="Pipeline stops here. No message sent, no claim filed — a human approves via the service layer.",
            details=["external_action_taken: False"],
        ),
    ]


class AgentState(TypedDict, total=False):
    shipment: dict
    extractions: list[dict]
    delay_hours: float | None
    document_mismatches: list[dict]
    classification: dict
    rule_classification: dict
    llm_classification: dict | None
    cross_check: dict | None
    classification_suggestion: dict | None
    policies: list[dict]
    draft: dict
    validation: dict
    approval_status: str


def build_graph(
    backend: ModelBackend | None = None,
    retriever: Retriever | None = None,
):
    """Compile the LangGraph pipeline with injectable backend + retriever."""
    backend = backend or MockModelBackend()
    retriever = retriever or KeywordRetriever()

    def extract(state: AgentState) -> AgentState:
        shipment = ShipmentInput.model_validate(state["shipment"])
        extractions = extract_documents(shipment, backend)
        return {"extractions": [e.model_dump() for e in extractions]}

    def ingest(state: AgentState) -> AgentState:
        shipment = ShipmentInput.model_validate(state["shipment"])
        return {
            "delay_hours": compute_delay_hours(
                shipment.scheduled_delivery, shipment.estimated_delivery
            ),
            "document_mismatches": [
                m.model_dump() for m in compare_documents(shipment.documents)
            ],
        }

    def classify(state: AgentState) -> AgentState:
        shipment = ShipmentInput.model_validate(state["shipment"])
        mismatches = compare_documents(shipment.documents)
        rule_result = classify_shipment(shipment, mismatches)
        llm = _llm_classify(state, shipment)
        if llm is _NO_LLM_BACKEND:
            return {
                "classification": rule_result.model_dump(mode="json"),
                "rule_classification": rule_result.model_dump(mode="json"),
                "llm_classification": None,
                "cross_check": None,
                "classification_suggestion": None,
            }
        final, cross = resolve_classification(rule_result, llm, getattr(backend, "name", ""))
        suggestion = None
        if llm:
            suggestion = dict(llm)
            suggestion["agrees_with_rules"] = (
                llm["exception_type"] == rule_result.exception_type.value
            )
        return {
            "classification": final.model_dump(mode="json"),
            "rule_classification": rule_result.model_dump(mode="json"),
            "llm_classification": llm,
            "cross_check": cross.model_dump(),
            "classification_suggestion": suggestion,
        }

    def _llm_classify(state: AgentState, shipment: ShipmentInput):
        """The LLM half of the cross-check — LLM backends only.

        Returns the ``_NO_LLM_BACKEND`` sentinel when the backend has no
        LLM classification (default mode: no cross-check at all), a parsed
        classification dict in provider mode, or ``None`` when the
        provider call failed or its reply was unusable — the cross-check
        then records ``rules_only`` and the run continues.
        """
        classify_fn = getattr(backend, "classify_with_llm", None)
        if classify_fn is None:
            return _NO_LLM_BACKEND
        context = DraftContext(
            shipment_id=shipment.shipment_id,
            origin=shipment.origin,
            destination=shipment.destination,
            carrier=shipment.carrier,
            status=shipment.status,
            latest_event=shipment.latest_event,
            condition_notes=shipment.condition_notes,
            delay_hours=state.get("delay_hours"),
            mismatches=state.get("document_mismatches", []),
        )
        try:
            return classify_fn(context)
        except Exception:  # the cross-check never fails the run
            return None

    def retrieve(state: AgentState) -> AgentState:
        classification = state["classification"]
        query = (
            f"{classification['exception_type']} {classification['rationale']} "
            f"customer update claim packet escalation"
        )
        policies = retriever.retrieve(query, top_k=3)
        return {"policies": [p.model_dump() for p in policies]}

    def draft(state: AgentState) -> AgentState:
        shipment = ShipmentInput.model_validate(state["shipment"])
        classification = state["classification"]
        citations = [p["policy_id"] for p in state.get("policies", [])]
        context = DraftContext(
            shipment_id=shipment.shipment_id,
            customer_name=shipment.customer_name,
            origin=shipment.origin,
            destination=shipment.destination,
            carrier=shipment.carrier,
            exception_type=classification["exception_type"],
            severity=classification["severity"],
            rationale=classification["rationale"],
            signals=classification["signals"],
            latest_event=shipment.latest_event,
            condition_notes=shipment.condition_notes,
            delay_hours=state.get("delay_hours"),
            mismatches=state.get("document_mismatches", []),
            citations=citations,
            policy_details=state.get("policies", []),
        )
        subject, body = backend.draft_customer_update(context)
        claim_packet = {
            "shipment_id": shipment.shipment_id,
            "exception_type": classification["exception_type"],
            "severity": classification["severity"],
            "document_mismatches": state.get("document_mismatches", []),
            "supporting_documents": [d.document_id for d in shipment.documents],
            "policy_citations": citations,
            "status": "draft — not filed",
        }
        draft_output = DraftOutput(
            subject=subject, body=body, claim_packet=claim_packet, citations=citations
        )
        return {"draft": draft_output.model_dump()}

    def validate(state: AgentState) -> AgentState:
        from .schemas import ExceptionType

        result = validate_draft(
            body=state["draft"]["body"],
            shipment_id=state["shipment"]["shipment_id"],
            exception_type=ExceptionType(state["classification"]["exception_type"]),
            citations=state["draft"]["citations"],
        )
        return {"validation": result.model_dump()}

    def human_approval(state: AgentState) -> AgentState:
        # The gate. Nothing leaves the system from here — a human must
        # explicitly approve via the API/CLI service layer.
        return {"approval_status": "awaiting_approval"}

    graph = StateGraph(AgentState)
    graph.add_node("extract", extract)
    graph.add_node("ingest", ingest)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("draft", draft)
    graph.add_node("validate", validate)
    graph.add_node("human_approval", human_approval)
    graph.set_entry_point("extract")
    for source, target in [
        ("extract", "ingest"),
        ("ingest", "classify"),
        ("classify", "retrieve"),
        ("retrieve", "draft"),
        ("draft", "validate"),
        ("validate", "human_approval"),
        ("human_approval", END),
    ]:
        graph.add_edge(source, target)
    return graph.compile()


def run_shipment(
    shipment: ShipmentInput | dict,
    backend: ModelBackend | None = None,
    retriever: Retriever | None = None,
) -> AgentResult:
    """Run one shipment through the full graph and return the typed result."""
    shipment_model = (
        shipment if isinstance(shipment, ShipmentInput) else ShipmentInput.model_validate(shipment)
    )
    app = build_graph(backend=backend, retriever=retriever)
    final = app.invoke({"shipment": shipment_model.model_dump(mode="json")})
    return AgentResult(
        shipment_id=shipment_model.shipment_id,
        classification=final["classification"],
        llm_suggestion=final.get("classification_suggestion"),
        cross_check=final.get("cross_check"),
        extractions=final.get("extractions", []),
        delay_hours=final.get("delay_hours"),
        document_mismatches=final.get("document_mismatches", []),
        policies=final.get("policies", []),
        draft=final["draft"],
        validation=final["validation"],
        trace=_build_trace(shipment_model, final),
        approval_status=final.get("approval_status", "awaiting_approval"),
        external_action_taken=False,
    )
