"""The agent graph (LangGraph).

Pipeline — every step is a named, inspectable node:

    extract ──► ingest ──► classify ──► retrieve ──► diagnose ──► options
        ──► draft ──► verify ──► review ──► validate ──► human_approval
        ──► END

The graph deliberately has NO node that sends a message, files a claim, or
touches an external system. It ends at the human-approval gate. Acting on an
approved draft is a separate, explicit step (see ``service.py``), which in
this prototype still performs no external action.
"""

from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .autonomy import compute_autonomy
from .clarify import build_information_request
from .classifier import classify_shipment
from .crosscheck import resolve_classification
from .diagnosis import build_diagnosis
from .extractor import (
    discrepancies_from_dicts,
    extract_documents,
    extraction_discrepancies,
)
from .guardrails import validate_draft
from .model_backends import DraftContext, ModelBackend, MockModelBackend, estimate_cost_usd
from .options import build_recovery_options
from .retriever import KeywordRetriever, Retriever
from .reviewer import review_draft
from .config import env_int, env_str, load_dotenv
from .schemas import (
    AgentResult,
    DocumentExtraction,
    DraftOutput,
    ExceptionType,
    ReviewerResult,
    RunTelemetry,
    ShipmentInput,
    TraceStep,
    ValidationResult,
    VerificationResult,
)
from .tools import compare_documents, compute_delay_hours, document_pair_warning
from .tools_agent import DiagnosisToolBox
from .verify import verify_draft

# Sentinel distinguishing "no LLM backend configured" (default mode — no
# cross-check is recorded at all) from "LLM configured but its reply was
# unusable" (cross-check records rules_only). See crosscheck.py for the
# resolution policy itself.
_NO_LLM_BACKEND = object()


def _retrieve_details(final: dict, policies: list[dict]) -> list[str]:
    """Retrieve-step details: the tool call, the hybrid merge → rerank
    steps when they ran, then the cited policies."""
    info = final.get("retrieval_info", {})
    mode = info.get("mode", "keyword")
    details = [
        f"tool call: search_policies(query, mode={mode}) -> {len(policies)} cited"
    ]
    if info.get("vector_store"):
        details.append(
            f"vector store: {info['vector_store']}"
            + (
                " (local Chroma persistent store)"
                if info["vector_store"] == "chroma"
                else " (in-memory cosine fallback — install the vectordb extra for Chroma)"
            )
        )
    if mode == "hybrid":
        details.append(
            f"merge: keyword pool {info.get('keyword_pool', 0)} + semantic pool "
            f"{info.get('semantic_pool', 0)}, deduped by policy ID"
        )
        details.append(
            "rerank: reciprocal-rank score fusion over the merged pool "
            "(score-fusion reranking in code — no cross-encoder)"
        )
    details += [
        f"[{p['policy_id']}] {p['title']} (score {p['score']}"
        + (f", via {p['retrieval']}" if p.get("retrieval") else "")
        + f"): {p['snippet']}"
        for p in policies
    ]
    return details


def _repair_settings() -> tuple[bool, int]:
    """Bounded-repair configuration (env, read at run time).

    ``GUARDRAIL_REPAIR`` — on by default; ``off``/``0``/``false``/``no``
    disables the repair loop entirely (a failed draft then behaves
    exactly as a no-repair pipeline: it stays failed and approval is
    refused). ``GUARDRAIL_REPAIR_MAX_ATTEMPTS`` — redraft attempts
    allowed per run, default 1, hard-capped at 3 so a misconfigured
    value cannot turn the loop unbounded.
    """
    load_dotenv()
    raw = (env_str("GUARDRAIL_REPAIR") or "on").strip().lower()
    enabled = raw not in {"off", "0", "false", "no"}
    attempts = min(max(env_int("GUARDRAIL_REPAIR_MAX_ATTEMPTS", 1), 0), 3)
    return enabled, attempts


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
    if final.get("classify_note"):
        classify_details.append(f"llm provider error: {final['classify_note']}")
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
    verification = final.get("verification") or {
        "grounded": True,
        "issues": [],
        "source": "checklist",
        "summary": "",
        "note": "",
    }
    autonomy = final.get("autonomy")
    review = final.get("review")
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
        if extraction.get("note"):
            extract_details.append(
                f"{extraction['document_id']}: fallback — {extraction['note']}"
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
    pair_warning = final.get("document_check_warning")
    if pair_warning:
        ingest_details.append(f"warning: {pair_warning}")
    for m in mismatches:
        ingest_details.append(
            f"mismatch — {m['field']}: BOL={m.get('bol_value')} vs invoice={m.get('invoice_value')}"
        )
    if not mismatches and not pair_warning:
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
            summary=(
                f"{len(policies)} policy snippet(s) retrieved from the synthetic SOP corpus "
                f"(mode: {final.get('retrieval_info', {}).get('mode', 'keyword')})."
            ),
            details=_retrieve_details(final, policies),
        ),
        TraceStep(
            name="diagnose",
            title="Diagnose root cause",
            summary=final["diagnosis"]["summary"],
            details=[
                f"root cause: {final['diagnosis']['root_cause']}",
                f"composition: {'LLM over the computed evidence' if final['diagnosis']['source'] == 'llm' else 'deterministic template over the computed evidence'}",
            ]
            + [
                f"tool call: {t['name']} -> {t['summary']}"
                for t in final["diagnosis"].get("tool_calls", [])
            ]
            + (
                [
                    "tool loop: provider-only — in the default mode the "
                    "deterministic evidence above is used directly"
                ]
                if final["diagnosis"]["source"] != "llm"
                else []
            )
            + ([f"fallback: {final['diagnosis']['note']}"] if final["diagnosis"].get("note") else [])
            + [f"evidence: {e}" for e in final["diagnosis"]["evidence"]],
        ),
        TraceStep(
            name="options",
            title="Recovery options (scored by code)",
            summary=(
                f"{len(final.get('recovery_options', []))} option(s) proposed; every score "
                "computed by deterministic code — recommended: "
                + next(
                    (o["title"] for o in final.get("recovery_options", []) if o["recommended"]),
                    "none",
                )
            ),
            details=[
                f"{o['option_id']} [{o['kind']}] {o['title']} — score {o['score']} "
                f"(ETA +{o['eta_improvement_hours']}h · added cost {o['added_cost_units']} units · "
                f"SLA {o['sla_score']})" + (" <- recommended" if o["recommended"] else "")
                for o in final.get("recovery_options", [])
            ]
            + [f"fallback: {n}" for n in final.get("options_notes", [])],
        ),
        TraceStep(
            name="draft",
            title="Draft update & claim packet",
            summary=f"Customer update drafted with {len(final['draft']['citations'])} policy citation(s); claim packet assembled as draft — not filed.",
            details=[f"subject: {final['draft']['subject']}"],
        ),
        TraceStep(
            name="verify",
            title="Self-verification — the agent critiques its own draft",
            status="passed" if verification["grounded"] else "failed",
            summary=(
                f"Draft checked against the verified facts and cited policies "
                f"({'LLM critique' if verification['source'] == 'llm' else 'deterministic checklist'}): "
                f"{'grounded' if verification['grounded'] else 'NOT grounded'}."
            ),
            details=[f"verdict: {'grounded' if verification['grounded'] else 'not grounded'}"]
            + ([f"summary: {verification['summary']}"] if verification.get("summary") else [])
            + [f"issue: {i}" for i in verification.get("issues", [])]
            + ([f"note: {verification['note']}"] if verification.get("note") else []),
        ),
        TraceStep(
            name="review",
            title="Independent review — a second pair of eyes",
            status=(
                "failed"
                if review and review["verdict"] == "block"
                else "passed"
                if review and review["verdict"] == "pass"
                else "completed"
            ),
            summary=(
                "Independent review disabled (REVIEWER=off) — no second read ran."
                if review is None
                else (
                    f"A reviewer that did not draft this update read it adversarially "
                    f"({'LLM reviewer' if review['source'] == 'llm' else 'deterministic checklist'}): "
                    f"verdict {review['verdict'].upper()}."
                    + (
                        " A block flags the case and disqualifies auto-approval — "
                        "the human still decides."
                        if review["verdict"] == "block"
                        else ""
                    )
                )
            ),
            details=(
                ["reviewer: off — REVIEWER=off; no independent review ran"]
                if review is None
                else [f"verdict: {review['verdict']}"]
                + ([f"model: {review['model']}"] if review.get("model") else [])
                + [f"finding: {f}" for f in review.get("findings", [])]
                + ([f"note: {review['note']}"] if review.get("note") else [])
            ),
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
            + [f"warning: {w}" for w in validation.get("warnings", [])]
            + (
                [
                    f"repair: bounded repair attempted ({final.get('repair_attempts', 0)} attempt(s)) — "
                    + (
                        "redraft passed the guardrails"
                        if final.get("repaired")
                        else "redraft still failed; original failure preserved on the result"
                    ),
                    *[
                        f"original failure: {e}"
                        for e in (final.get("original_validation") or {}).get("errors", [])
                    ],
                ]
                if final.get("repair_attempted")
                else []
            ),
        ),
        TraceStep(
            name="human_approval",
            title="Human approval gate",
            status="awaiting",
            summary="Pipeline stops here. No message sent, no claim filed — a human approves via the service layer.",
            details=["external_action_taken: False"]
            + (
                [
                    "autonomy recommendation: "
                    + (
                        "eligible for auto-approval"
                        if autonomy["eligible_for_auto_approval"]
                        else "human decision required"
                    )
                    + " (recommendation only — the gate is unchanged)",
                    *[f"autonomy: {r}" for r in autonomy.get("reasons", [])],
                ]
                if autonomy
                else []
            )
            + (
                [
                    "information requested: this case is under-determined — a "
                    "clarification request was composed for the carrier/ops "
                    "contact and attached to the result (not sent)",
                    *[
                        f"missing: {item}"
                        for item in (final.get("information_request") or {}).get(
                            "missing_items", []
                        )
                    ],
                ]
                if final.get("needs_information")
                else []
            ),
        ),
    ]


class AgentState(TypedDict, total=False):
    shipment: dict
    extractions: list[dict]
    delay_hours: float | None
    document_mismatches: list[dict]
    document_check_warning: str | None
    classification: dict
    rule_classification: dict
    llm_classification: dict | None
    classify_note: str | None
    cross_check: dict | None
    classification_suggestion: dict | None
    policies: list[dict]
    retrieval_info: dict
    history: dict | None
    priors: list[dict]
    diagnosis: dict
    recovery_options: list[dict]
    options_notes: list[str]
    recommended_option_id: str | None
    draft: dict
    verification: dict
    review: dict | None
    reviewer_blocked: bool
    validation: dict
    repair_attempted: bool
    repaired: bool
    repair_attempts: int
    original_validation: dict | None
    autonomy: dict
    needs_information: bool
    information_request: dict | None
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
            "document_check_warning": document_pair_warning(shipment.documents),
        }

    def classify(state: AgentState) -> AgentState:
        shipment = ShipmentInput.model_validate(state["shipment"])
        mismatches = compare_documents(shipment.documents)
        rule_result = classify_shipment(shipment, mismatches)
        llm, classify_note = _llm_classify(state, shipment)
        if llm is _NO_LLM_BACKEND:
            return {
                "classification": rule_result.model_dump(mode="json"),
                "rule_classification": rule_result.model_dump(mode="json"),
                "llm_classification": None,
                "classify_note": None,
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
            "classify_note": classify_note,
            "cross_check": cross.model_dump(),
            "classification_suggestion": suggestion,
        }

    def _llm_classify(state: AgentState, shipment: ShipmentInput):
        """The LLM half of the cross-check — LLM backends only.

        Returns ``(value, error_note)``. ``value`` is the
        ``_NO_LLM_BACKEND`` sentinel when the backend has no LLM
        classification (default mode: no cross-check at all), a parsed
        classification dict in provider mode, or ``None`` when the
        provider call failed or its reply was unusable — the cross-check
        then records ``rules_only`` and the run continues. A provider
        failure also returns its translated message as ``error_note``
        so the trace records the degradation instead of hiding it.
        """
        classify_fn = getattr(backend, "classify_with_llm", None)
        if classify_fn is None:
            return _NO_LLM_BACKEND, None
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
            return classify_fn(context), None
        except Exception as exc:  # the cross-check never fails the run
            return None, str(exc)

    def retrieve(state: AgentState) -> AgentState:
        classification = state["classification"]
        shipment = ShipmentInput.model_validate(state["shipment"])
        # The query is built from the shipment's OWN content — its latest
        # event text, condition notes, and document field values — not
        # just the exception type and the classifier's rationale. A
        # customer's SOP written in operational language ("cartons
        # crushed at terminal inspection") must rank for the case it
        # describes, even when its vocabulary shares nothing with the
        # rationale's wording.
        doc_values = [
            str(value)
            for document in shipment.documents
            for value in document.fields.values()
        ]
        query = " ".join(
            part
            for part in [
                classification["exception_type"],
                classification["rationale"],
                shipment.latest_event,
                shipment.condition_notes,
                *doc_values,
                "customer update claim packet escalation",
            ]
            if part
        )
        policies = retriever.retrieve(query, top_k=3)
        info = {"mode": getattr(retriever, "name", "keyword")}
        info.update(getattr(retriever, "last_stats", {}) or {})
        vector_store = getattr(retriever, "vector_store", None)
        if vector_store:
            info["vector_store"] = vector_store
        return {
            "policies": [p.model_dump() for p in policies],
            "retrieval_info": info,
        }

    def diagnose(state: AgentState) -> AgentState:
        shipment = ShipmentInput.model_validate(state["shipment"])
        extractions = [
            DocumentExtraction.model_validate(e) for e in state.get("extractions", [])
        ]
        # The toolbox is what makes the provider-mode diagnosis agentic:
        # the model can pull policy search, history, and computed facts
        # through it before composing. In the default mode nothing calls
        # it — the deterministic diagnosis runs on the same evidence.
        toolbox = DiagnosisToolBox(
            shipment=shipment,
            classification=state["classification"],
            delay_hours=state.get("delay_hours"),
            mismatches=state.get("document_mismatches", []),
            discrepancies=extraction_discrepancies(extractions),
            document_check_warning=state.get("document_check_warning"),
            retriever=retriever,
            priors=state.get("priors", []),
        )
        diagnosis = build_diagnosis(
            shipment=shipment,
            classification=state["classification"],
            delay_hours=state.get("delay_hours"),
            mismatches=state.get("document_mismatches", []),
            extractions=extractions,
            policies=state.get("policies", []),
            backend=backend,
            history=state.get("history"),
            toolbox=toolbox,
        )
        return {"diagnosis": diagnosis.model_dump()}

    def options(state: AgentState) -> AgentState:
        shipment = ShipmentInput.model_validate(state["shipment"])
        classification = state["classification"]
        context = DraftContext(
            shipment_id=shipment.shipment_id,
            origin=shipment.origin,
            destination=shipment.destination,
            carrier=shipment.carrier,
            exception_type=classification["exception_type"],
            severity=classification["severity"],
            delay_hours=state.get("delay_hours"),
            mismatches=state.get("document_mismatches", []),
            diagnosis_summary=state["diagnosis"]["summary"],
            policy_details=state.get("policies", []),
        )
        notes: list[str] = []
        scored = build_recovery_options(
            exception_type=classification["exception_type"],
            severity=classification["severity"],
            delay_hours=state.get("delay_hours"),
            backend=backend,
            context=context,
            notes=notes,
        )
        recommended = next((o for o in scored if o.recommended), None)
        return {
            "recovery_options": [o.model_dump() for o in scored],
            "recommended_option_id": recommended.option_id if recommended else None,
            "options_notes": notes,
        }

    def _build_draft(state: AgentState, repair_feedback: str = "") -> DraftOutput:
        shipment = ShipmentInput.model_validate(state["shipment"])
        classification = state["classification"]
        citations = [p["policy_id"] for p in state.get("policies", [])]
        recommended = next(
            (o for o in state.get("recovery_options", []) if o["recommended"]), None
        )
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
            diagnosis_summary=state["diagnosis"]["summary"],
            recommended_option_text=(
                f"{recommended['title']} — {recommended['description']}"
                if recommended
                else ""
            ),
            repair_feedback=repair_feedback,
        )
        subject, body = backend.draft_customer_update(context)
        claim_packet = {
            "shipment_id": shipment.shipment_id,
            "exception_type": classification["exception_type"],
            "severity": classification["severity"],
            "document_mismatches": state.get("document_mismatches", []),
            "supporting_documents": [d.document_id for d in shipment.documents],
            "policy_citations": citations,
            "diagnosis": state["diagnosis"],
            "recovery_options": state.get("recovery_options", []),
            "recommended_option_id": state.get("recommended_option_id"),
            "status": "draft — not filed",
        }
        return DraftOutput(
            subject=subject, body=body, claim_packet=claim_packet, citations=citations
        )

    def draft(state: AgentState) -> AgentState:
        return {"draft": _build_draft(state).model_dump()}

    def verify(state: AgentState) -> AgentState:
        draft = DraftOutput.model_validate(state["draft"])
        verification = verify_draft(
            draft=draft,
            classification=state["classification"],
            delay_hours=state.get("delay_hours"),
            mismatches=state.get("document_mismatches", []),
            policies=state.get("policies", []),
            shipment_id=state["shipment"]["shipment_id"],
            backend=backend,
        )
        # The verdict travels inside the claim packet too — the approver
        # (and any downstream system) sees the self-critique next to the
        # draft it judges.
        draft.claim_packet = {
            **draft.claim_packet,
            "verification": verification.model_dump(),
        }
        return {"verification": verification.model_dump(), "draft": draft.model_dump()}

    def _run_review(state: AgentState, draft: DraftOutput) -> ReviewerResult | None:
        """The independent review of one draft version (review node +
        the repair loop, which re-reviews every redraft)."""
        return review_draft(
            draft=draft,
            classification=state["classification"],
            diagnosis=state.get("diagnosis"),
            delay_hours=state.get("delay_hours"),
            mismatches=state.get("document_mismatches", []),
            policies=state.get("policies", []),
            backend=backend,
        )

    def review(state: AgentState) -> AgentState:
        # Generator/critic split: a reviewer that did not draft the
        # update reads it adversarially (reviewer.py). A block verdict
        # flags the result and disqualifies auto-approval — it never
        # rejects anything itself; the human still decides.
        draft = DraftOutput.model_validate(state["draft"])
        review_result = _run_review(state, draft)
        if review_result is not None:
            draft.claim_packet = {
                **draft.claim_packet,
                "review": review_result.model_dump(),
            }
        return {
            "review": review_result.model_dump() if review_result else None,
            "reviewer_blocked": bool(
                review_result and review_result.verdict == "block"
            ),
            "draft": draft.model_dump(),
        }

    def _run_guardrails(state: AgentState, draft: DraftOutput) -> ValidationResult:
        return validate_draft(
            body=draft.body,
            shipment_id=state["shipment"]["shipment_id"],
            exception_type=ExceptionType(state["classification"]["exception_type"]),
            citations=draft.citations,
        )

    def validate(state: AgentState) -> AgentState:
        """Guardrails, with a bounded repair loop on failure.

        On failure — and only when repair is enabled (GUARDRAIL_REPAIR,
        default on, GUARDRAIL_REPAIR_MAX_ATTEMPTS default 1) — the agent
        redrafts once with the guardrail failure reasons and the
        self-verification issues fed back into the drafting prompt, then
        re-verifies and re-validates. The original failure is preserved
        on the result (``original_validation``) and the attempt is
        flagged (``repair_attempted`` / ``repaired``); the guardrail
        rules themselves never change. With repair off, a failed draft
        stays failed, exactly as a no-repair pipeline behaves.
        """
        draft = DraftOutput.model_validate(state["draft"])
        validation = _run_guardrails(state, draft)
        update: AgentState = {
            "validation": validation.model_dump(),
            "repair_attempted": False,
            "repaired": False,
            "repair_attempts": 0,
            "original_validation": None,
        }
        if validation.passed:
            return update
        enabled, max_attempts = _repair_settings()
        if not enabled or max_attempts == 0:
            return update
        verification = (
            VerificationResult.model_validate(state["verification"])
            if state.get("verification")
            else None
        )
        feedback_lines = ["Guardrail failures to fix:"] + [
            f"- {error}" for error in validation.errors
        ]
        if verification is not None and verification.issues:
            feedback_lines.append("Self-verification issues to address:")
            feedback_lines += [f"- {issue}" for issue in verification.issues]
        feedback = "\n".join(feedback_lines)
        original = validation
        attempts = 0
        working = dict(state)
        review_result = (
            ReviewerResult.model_validate(state["review"])
            if state.get("review")
            else None
        )
        while attempts < max_attempts and not validation.passed:
            attempts += 1
            draft = _build_draft(working, repair_feedback=feedback)
            verification = verify_draft(
                draft=draft,
                classification=state["classification"],
                delay_hours=state.get("delay_hours"),
                mismatches=state.get("document_mismatches", []),
                policies=state.get("policies", []),
                shipment_id=state["shipment"]["shipment_id"],
                backend=backend,
            )
            draft.claim_packet = {
                **draft.claim_packet,
                "verification": verification.model_dump(),
            }
            # The reviewer judges the draft the approver will actually
            # see — so every redraft is re-reviewed, not just re-verified.
            review_result = _run_review(state, draft)
            if review_result is not None:
                draft.claim_packet = {
                    **draft.claim_packet,
                    "review": review_result.model_dump(),
                }
            validation = _run_guardrails(state, draft)
            working = {**working, "draft": draft.model_dump()}
        update.update(
            {
                "validation": validation.model_dump(),
                "draft": draft.model_dump(),
                "verification": verification.model_dump() if verification else None,
                "review": review_result.model_dump() if review_result else None,
                "reviewer_blocked": bool(
                    review_result and review_result.verdict == "block"
                ),
                "repair_attempted": attempts > 0,
                "repaired": validation.passed,
                "repair_attempts": attempts,
                "original_validation": original.model_dump(),
            }
        )
        return update

    def human_approval(state: AgentState) -> AgentState:
        # The gate. Nothing leaves the system from here — a human must
        # explicitly approve via the API/CLI service layer. The autonomy
        # recommendation computed here is printed on the result and the
        # claim packet; it never opens this gate by itself.
        autonomy = compute_autonomy(
            classification=state["classification"],
            validation=state["validation"],
            cross_check=state.get("cross_check"),
            repair_attempted=bool(state.get("repair_attempted")),
            reviewer_blocked=bool(state.get("reviewer_blocked")),
        )
        # Information-needed flow: an under-determined case (classification
        # "none" at low confidence, with concrete inputs missing) gets a
        # composed clarification request attached — the gate is unchanged
        # and nothing is sent (see clarify.py).
        shipment = ShipmentInput.model_validate(state["shipment"])
        info_request = build_information_request(
            shipment=shipment,
            classification=state["classification"],
            document_check_warning=state.get("document_check_warning"),
            extractions=state.get("extractions", []),
            backend=backend,
        )
        draft = DraftOutput.model_validate(state["draft"])
        draft.claim_packet = {
            **draft.claim_packet,
            "autonomy_recommendation": autonomy.model_dump(),
        }
        if info_request is not None:
            draft.claim_packet = {
                **draft.claim_packet,
                "information_request": info_request.model_dump(),
            }
        return {
            "approval_status": "awaiting_approval",
            "autonomy": autonomy.model_dump(),
            "needs_information": info_request is not None,
            "information_request": info_request.model_dump() if info_request else None,
            "draft": draft.model_dump(),
        }

    graph = StateGraph(AgentState)
    graph.add_node("extract", extract)
    graph.add_node("ingest", ingest)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("diagnose", diagnose)
    graph.add_node("options", options)
    graph.add_node("draft", draft)
    graph.add_node("verify", verify)
    graph.add_node("review", review)
    graph.add_node("validate", validate)
    graph.add_node("human_approval", human_approval)
    graph.set_entry_point("extract")
    for source, target in [
        ("extract", "ingest"),
        ("ingest", "classify"),
        ("classify", "retrieve"),
        ("retrieve", "diagnose"),
        ("diagnose", "options"),
        ("options", "draft"),
        ("draft", "verify"),
        ("verify", "review"),
        ("review", "validate"),
        ("validate", "human_approval"),
        ("human_approval", END),
    ]:
        graph.add_edge(source, target)
    return graph.compile()


def run_shipment(
    shipment: ShipmentInput | dict,
    backend: ModelBackend | None = None,
    retriever: Retriever | None = None,
    history: dict | None = None,
    priors: list[dict] | None = None,
) -> AgentResult:
    """Run one shipment through the full graph and return the typed result.

    ``history`` is the memory summary for this shipment's consignee and
    lane (see ``service.ShipmentService.analyze``); it reaches the
    diagnosis as evidence. ``priors`` is the raw stored-history entries
    behind that summary — the diagnosis tool loop reads them (lane and
    carrier history tools). Direct callers usually leave both None.
    """
    shipment_model = (
        shipment if isinstance(shipment, ShipmentInput) else ShipmentInput.model_validate(shipment)
    )
    effective_backend = backend or MockModelBackend()
    app = build_graph(backend=effective_backend, retriever=retriever)
    # Telemetry: snapshot the backend's cumulative usage around the run
    # (a service reuses one backend across runs, so the run's share is
    # the delta), and time the whole invoke on the wall clock.
    usage_totals = getattr(effective_backend, "usage_totals", None)
    before = usage_totals() if callable(usage_totals) else None
    started = time.perf_counter()
    final = app.invoke(
        {
            "shipment": shipment_model.model_dump(mode="json"),
            "history": history,
            "priors": priors or [],
        }
    )
    latency = round(time.perf_counter() - started, 3)
    telemetry = _run_telemetry(effective_backend, before, latency)
    result = AgentResult(
        shipment_id=shipment_model.shipment_id,
        classification=final["classification"],
        llm_suggestion=final.get("classification_suggestion"),
        cross_check=final.get("cross_check"),
        extractions=final.get("extractions", []),
        diagnosis=final.get("diagnosis"),
        recovery_options=final.get("recovery_options", []),
        recommended_option_id=final.get("recommended_option_id"),
        delay_hours=final.get("delay_hours"),
        document_mismatches=final.get("document_mismatches", []),
        document_check_warning=final.get("document_check_warning"),
        policies=final.get("policies", []),
        draft=final["draft"],
        verification=final.get("verification"),
        review=final.get("review"),
        reviewer_blocked=bool(final.get("reviewer_blocked")),
        validation=final["validation"],
        repair_attempted=bool(final.get("repair_attempted")),
        repaired=bool(final.get("repaired")),
        repair_attempts=int(final.get("repair_attempts") or 0),
        original_validation=final.get("original_validation"),
        autonomy=final.get("autonomy"),
        needs_information=bool(final.get("needs_information")),
        information_request=final.get("information_request"),
        telemetry=telemetry,
        trace=_build_trace(shipment_model, final),
        approval_status=final.get("approval_status", "awaiting_approval"),
        external_action_taken=False,
    )
    # The telemetry rides inside the claim packet too — whoever receives
    # the packet (the approver, a downstream system) sees what the run
    # that produced it cost.
    result.draft.claim_packet = {
        **result.draft.claim_packet,
        "telemetry": telemetry.model_dump(),
    }
    return result


def _run_telemetry(backend, before: dict | None, latency: float) -> RunTelemetry:
    """Aggregate one run's telemetry from the backend's usage counters.

    Provider mode: real token usage (the delta over the run), the model
    that served it, and the estimated cost from the price table (None
    for an unlisted model). Mock mode: tokens and cost are None — no
    model ran, and the telemetry says so instead of inventing numbers —
    while the call count and latency remain real.
    """
    usage_totals = getattr(backend, "usage_totals", None)
    after = usage_totals() if callable(usage_totals) else None
    backend_name = getattr(backend, "name", "mock")
    if before is None or after is None:
        return RunTelemetry(backend=backend_name, latency_seconds=latency)
    calls = max(int(after.get("calls", 0)) - int(before.get("calls", 0)), 0)
    if backend_name == "mock":
        return RunTelemetry(
            backend=backend_name, model_calls=calls, latency_seconds=latency
        )
    input_tokens = max(
        int(after.get("input_tokens", 0)) - int(before.get("input_tokens", 0)), 0
    )
    output_tokens = max(
        int(after.get("output_tokens", 0)) - int(before.get("output_tokens", 0)), 0
    )
    model = getattr(backend, "_model", None)
    return RunTelemetry(
        backend=backend_name,
        model=model,
        model_calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimate_cost_usd(model, input_tokens, output_tokens),
        latency_seconds=latency,
    )
