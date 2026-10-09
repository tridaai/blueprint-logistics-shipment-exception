"""The agent graph (LangGraph).

Pipeline — every step is a named, inspectable node:

    ingest ──► classify ──► retrieve ──► draft ──► validate ──► human_approval ──► END

The graph deliberately has NO node that sends a message, files a claim, or
touches an external system. It ends at the human-approval gate. Acting on an
approved draft is a separate, explicit step (see ``service.py``), which in
this prototype still performs no external action.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .classifier import classify_shipment
from .guardrails import validate_draft
from .model_backends import DraftContext, ModelBackend, MockModelBackend
from .retriever import KeywordRetriever, Retriever
from .schemas import AgentResult, DraftOutput, ShipmentInput, TraceStep
from .tools import compare_documents, compute_delay_hours


def _build_trace(shipment: ShipmentInput, final: dict) -> list[TraceStep]:
    """Assemble the inspectable per-step trace from the final graph state."""
    classification = final["classification"]
    mismatches = final.get("document_mismatches", [])
    delay = final.get("delay_hours")
    policies = final.get("policies", [])
    validation = final["validation"]

    ingest_details = [f"documents compared: {len(shipment.documents)}"]
    if delay is not None:
        ingest_details.append(f"delay vs schedule computed: {delay} hours")
    for m in mismatches:
        ingest_details.append(
            f"mismatch — {m['field']}: BOL={m.get('bol_value')} vs invoice={m.get('invoice_value')}"
        )
    if not mismatches:
        ingest_details.append("no document mismatches found")

    return [
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
            details=[f"evidence: {s}" for s in classification.get("signals", [])] or ["evidence: (no signals)"],
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
    delay_hours: float | None
    document_mismatches: list[dict]
    classification: dict
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
        result = classify_shipment(shipment, mismatches)
        return {"classification": result.model_dump(mode="json")}

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
    graph.add_node("ingest", ingest)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("draft", draft)
    graph.add_node("validate", validate)
    graph.add_node("human_approval", human_approval)
    graph.set_entry_point("ingest")
    for source, target in [
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
        delay_hours=final.get("delay_hours"),
        document_mismatches=final.get("document_mismatches", []),
        policies=final.get("policies", []),
        draft=final["draft"],
        validation=final["validation"],
        trace=_build_trace(shipment_model, final),
        approval_status=final.get("approval_status", "awaiting_approval"),
        external_action_taken=False,
    )
