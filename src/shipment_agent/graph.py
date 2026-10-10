"""The agent graph (LangGraph).

Pipeline — every step is a named, inspectable node. The evidence
phase fans out (independent branches run concurrently and merge
before diagnose); the rest is a chain:

    extract ──┐
              ├──► classify ──┬──► retrieve ──────┐
    ingest ───┘               ├──► discrepancies ──┼──► diagnose ──► options
                              └──► history ────────┘
        ──► draft ──► verify ──► review ──► validate ──► human_approval
        ──► approval_gate ──► END

The graph deliberately has NO node that sends a message, files a claim, or
touches an external system. It ends at the human-approval gate: with a
checkpointer attached (see ``checkpoints.py``), ``approval_gate`` is a
real ``interrupt()`` — the run's state persists there and the service's
approve/reject resumes the thread with the decision. Without a
checkpointer the gate node is a pass-through and the flow is exactly
the service-state flow. Acting on an approved draft is a separate,
explicit step (see ``service.py``), which in this prototype still
performs no external action.
"""

from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .autonomy import compute_autonomy
from .clarify import build_information_request
from .classifier import classify_shipment
from .crosscheck import resolve_classification
from .diagnosis import build_diagnosis, memory_evidence_lines
from .extractor import (
    discrepancies_from_dicts,
    extract_documents,
    extraction_discrepancies,
)
from .guardrails import validate_draft
from .model_backends import DraftContext, ModelBackend, MockModelBackend, estimate_cost_usd
from .tracing import node_span, run_span
from .options import build_recovery_options
from .retriever import KeywordRetriever, Retriever
from .reviewer import review_draft
from .screening import sanitized_event_notes, screen_shipment
from .config import env_int, env_str, load_dotenv
from .schemas import (
    AgentResult,
    DocumentExtraction,
    DraftOutput,
    ExceptionType,
    ReviewerResult,
    RunTelemetry,
    ShipmentInput,
    TokenBudget,
    TraceStep,
    ValidationResult,
    VerificationResult,
)
from .events import RunEvent
from .ports import Checkpointer, EventSink
from .resilience import ResilientBackend
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


def _token_budget() -> int | None:
    """``RUN_TOKEN_BUDGET`` (env, read at run time): the per-run token
    ceiling for provider calls. Unset, zero, or invalid = off."""
    load_dotenv()
    limit = env_int("RUN_TOKEN_BUDGET", 0)
    return limit if limit > 0 else None


# Provider capability method -> the pipeline node it serves. When the
# budget is spent, calls to these methods return None, so each node's
# existing deterministic fallback engages exactly as it does when a
# provider reply is unusable.
_PROVIDER_STEP_NODES = {
    "extract_document_fields": "extract",
    "classify_with_llm": "classify",
    "diagnose": "diagnose",
    "diagnose_with_tools": "diagnose",
    "propose_options": "options",
    "draft_customer_update": "draft",
    "verify_draft": "verify",
    "review_draft": "review",
    "compose_information_request": "human_approval",
}


class _BudgetGuard:
    """Backend wrapper enforcing the per-run token budget (cost guardrail).

    Wraps the run's backend for one graph invocation. Every provider
    capability call re-checks the run's cumulative provider tokens
    (the backend's own usage counters) at call time; once the budget
    is exceeded, the call returns None — the node's deterministic /
    template path serves the step, the degradation is recorded in
    ``degraded_nodes`` for the trace, and the run continues. Drafting,
    which has no in-node fallback, renders through the deterministic
    template backend instead. The budget never hard-fails a run.
    """

    def __init__(self, backend, limit: int) -> None:
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_limit", limit)
        object.__setattr__(self, "_start_tokens", self._tokens_of(backend))
        object.__setattr__(self, "degraded_nodes", [])

    @staticmethod
    def _tokens_of(backend) -> int:
        totals_fn = getattr(backend, "usage_totals", None)
        if not callable(totals_fn):
            return 0
        totals = totals_fn()
        return int(totals.get("input_tokens", 0)) + int(totals.get("output_tokens", 0))

    def used_tokens(self) -> int:
        return self._tokens_of(self._backend) - self._start_tokens

    def exceeded(self) -> bool:
        return self.used_tokens() > self._limit

    def status(self) -> dict:
        return {
            "limit": self._limit,
            "used": self.used_tokens(),
            "exceeded": self.exceeded(),
            "degraded_nodes": list(self.degraded_nodes),
        }

    def _note(self, method: str) -> None:
        node = _PROVIDER_STEP_NODES[method]
        if node not in self.degraded_nodes:
            self.degraded_nodes.append(node)

    def __getattr__(self, name: str):
        backend = object.__getattribute__(self, "_backend")
        attr = getattr(backend, name)  # AttributeError propagates — getattr defaults work
        if name not in _PROVIDER_STEP_NODES or not callable(attr):
            return attr

        def guarded(*args, **kwargs):
            # Checked at call time, not access time: some nodes fetch a
            # method once and call it repeatedly (extraction, per
            # document) — the budget must bite mid-list too.
            if self.exceeded():
                self._note(name)
                return None
            return attr(*args, **kwargs)

        return guarded

    def draft_customer_update(self, context):
        """Drafting cannot degrade to "no method" — over budget it
        renders through the deterministic template backend instead."""
        if self.exceeded():
            self._note("draft_customer_update")
            return MockModelBackend().draft_customer_update(context)
        return self._backend.draft_customer_update(context)


def _build_trace(
    shipment: ShipmentInput,
    final: dict,
    durations: dict[str, float] | None = None,
    retry_notes: list[dict] | None = None,
) -> list[TraceStep]:
    """Assemble the inspectable per-step trace from the final graph state.

    ``durations`` maps node name → wall-clock milliseconds, measured by
    the node wrappers in :func:`build_graph` (the same measurement the
    run-event stream reports); each step carries its node's duration.
    ``retry_notes`` are the resilience wrapper's retry records — each
    lands on its node's step ("attempt 2 after provider error").
    """
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
    injection_flags = final.get("injection_flags", [])
    if injection_flags:
        for flag in injection_flags:
            ingest_details.append(
                f"injection screen: FLAG in {flag['field']} — pattern "
                f"'{flag['pattern']}': \"{flag['excerpt']}\" (flagged sentences "
                "are kept out of drafts and prompts)"
            )
    else:
        ingest_details.append(
            "injection screen: no instruction-like content found in the untrusted fields"
        )
    for m in mismatches:
        ingest_details.append(
            f"mismatch — {m['field']}: BOL={m.get('bol_value')} vs invoice={m.get('invoice_value')}"
        )
    if not mismatches and not pair_warning:
        ingest_details.append("no document mismatches found")

    steps = [
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
    if durations:
        for step in steps:
            if step.name in durations:
                step.duration_ms = durations[step.name]
    if retry_notes:
        for note in retry_notes:
            for step in steps:
                if step.name == note["node"]:
                    step.details.append(
                        f"retry: attempt {note['attempt']} after provider "
                        f"error ({note['error']})"
                    )
    budget = final.get("token_budget")
    if budget:
        degraded = budget.get("degraded_nodes", [])
        for step in steps:
            if step.name in degraded:
                step.details.append(
                    "budget exceeded — template path (RUN_TOKEN_BUDGET spent; "
                    "this step was served by its deterministic path)"
                )
            if step.name == "human_approval":
                step.details.append(
                    f"token budget: {budget['used']} of {budget['limit']} tokens used"
                    + (
                        " — budget exceeded; provider steps degraded to their "
                        "template paths: " + ", ".join(degraded)
                        if budget.get("exceeded")
                        else " — within budget"
                    )
                )
    return steps


class AgentState(TypedDict, total=False):
    shipment: dict
    extractions: list[dict]
    injection_flags: list[dict]
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
    extraction_discrepancies: list[str]
    history_evidence: list[str]
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
    token_budget: dict | None
    approval_status: str


def build_graph(
    backend: ModelBackend | None = None,
    retriever: Retriever | None = None,
    event_sink: EventSink | None = None,
    run_context: dict | None = None,
    checkpointer: Checkpointer | None = None,
    evidence_mode: str = "parallel",
):
    """Compile the LangGraph pipeline with injectable backend + retriever.

    ``event_sink`` (a ports.EventSink) receives the structured run
    events the node wrappers emit — node start/finish with durations,
    plus the semantic events derived from node updates (tool calls,
    the guardrail verdict, repair attempts). ``run_context``, when
    given, is filled with the run's measurements (``durations``:
    node name → milliseconds) for the caller to fold into the trace.

    ``checkpointer`` (a ports.Checkpointer), when given, turns the
    final node into a checkpointed approval gate: the run pauses at
    an ``interrupt()`` with its state persisted under the caller's
    thread_id, and :func:`resume_approval` completes it with the
    human's decision. Without one, the gate node is a pass-through.

    ``evidence_mode`` selects the evidence-phase topology:
    ``"parallel"`` (default) fans the independent work out — extract
    ∥ ingest, then retrieval ∥ extraction cross-check ∥ history
    evidence after classification, merging before diagnose — while
    ``"sequential"`` chains the same node functions linearly. The
    node functions and the merge are identical; only the schedule
    differs, and an equivalence test pins that the results are too.
    """
    if evidence_mode not in ("parallel", "sequential"):
        raise ValueError(
            f"evidence_mode must be 'parallel' or 'sequential', got {evidence_mode!r}"
        )
    backend = backend or MockModelBackend()
    retriever = retriever or KeywordRetriever()
    # Cost guardrail: with RUN_TOKEN_BUDGET set, the backend is wrapped
    # for this run — once the run's provider tokens pass the budget, the
    # remaining provider steps degrade to their deterministic paths
    # (see _BudgetGuard). Unset, the backend runs unwrapped, unchanged.
    budget_limit = _token_budget()
    budget_guard = _BudgetGuard(backend, budget_limit) if budget_limit else None
    if budget_guard is not None:
        backend = budget_guard
    # Resilience policy (resilience.py): per-call timeout for every
    # provider call, one retry for the idempotent language steps.
    # Wrapped OUTSIDE the budget guard, so each retry attempt
    # re-checks the budget before spending.
    resilient = ResilientBackend(backend)
    backend = resilient
    if run_context is not None:
        run_context["retry_log"] = resilient.retry_log

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
            # Prompt-injection screen over the untrusted fields. The
            # flags are evidence for the approver; the flagged sentences
            # are kept out of drafts and prompts by the sanitisation at
            # the construction sites (screening.py).
            "injection_flags": [
                flag.model_dump() for flag in screen_shipment(shipment)
            ],
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
        event_text, notes_text = sanitized_event_notes(shipment)
        context = DraftContext(
            shipment_id=shipment.shipment_id,
            origin=shipment.origin,
            destination=shipment.destination,
            carrier=shipment.carrier,
            status=shipment.status,
            latest_event=event_text,
            condition_notes=notes_text,
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

    def discrepancies(state: AgentState) -> AgentState:
        # Evidence fan-out branch: the extraction cross-check (provided
        # vs extracted field comparison), computed from the extraction
        # records. Pure and independent of retrieval/history, so it
        # runs concurrently with them (see the topology below).
        extractions = [
            DocumentExtraction.model_validate(e) for e in state.get("extractions", [])
        ]
        return {"extraction_discrepancies": extraction_discrepancies(extractions)}

    def history(state: AgentState) -> AgentState:
        # Evidence fan-out branch: the memory + reviewer-feedback
        # evidence lines over the store summary the service supplied.
        return {"history_evidence": memory_evidence_lines(state.get("history"))}

    def diagnose(state: AgentState) -> AgentState:
        shipment = ShipmentInput.model_validate(state["shipment"])
        extractions = [
            DocumentExtraction.model_validate(e) for e in state.get("extractions", [])
        ]
        # The fan-out branches computed the cross-check discrepancies
        # and the history evidence in parallel; fall back to computing
        # them here if a branch did not run (defensive — the topology
        # guarantees both precede this node).
        disc = state.get("extraction_discrepancies")
        if disc is None:
            disc = extraction_discrepancies(extractions)
        # The toolbox is what makes the provider-mode diagnosis agentic:
        # the model can pull policy search, history, and computed facts
        # through it before composing. In the default mode nothing calls
        # it — the deterministic diagnosis runs on the same evidence.
        toolbox = DiagnosisToolBox(
            shipment=shipment,
            classification=state["classification"],
            delay_hours=state.get("delay_hours"),
            mismatches=state.get("document_mismatches", []),
            discrepancies=disc,
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
            discrepancies=disc,
            history_lines=state.get("history_evidence"),
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
        # Scorecard-aware scoring: the service put the carrier's
        # track record and the fleet baseline in the run's history;
        # the scorer's reliability term reads them (options.py). The
        # lane figures ride along too — when the carrier's history
        # on this lane is thick enough, they are the figures the
        # term reads. A direct run (evals, demos) has none of them —
        # base scores stand.
        history = state.get("history") or {}
        scored = build_recovery_options(
            exception_type=classification["exception_type"],
            severity=classification["severity"],
            delay_hours=state.get("delay_hours"),
            backend=backend,
            context=context,
            notes=notes,
            carrier_scorecard=history.get("carrier_scorecard"),
            fleet_baseline=history.get("fleet_baseline"),
            carrier_lane_scorecard=history.get("carrier_lane_scorecard"),
            lane_baseline=history.get("lane_baseline"),
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
        event_text, notes_text = sanitized_event_notes(shipment)
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
            latest_event=event_text,
            condition_notes=notes_text,
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
        if state.get("injection_flags"):
            # The approver sees that the input tried something — the
            # flags ride in the packet next to the draft they protect.
            draft.claim_packet = {
                **draft.claim_packet,
                "injection_flags": state["injection_flags"],
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
            "token_budget": budget_guard.status() if budget_guard is not None else None,
            "draft": draft.model_dump(),
        }

    def approval_gate(state: AgentState) -> AgentState:
        # The checkpointed gate. With a checkpointer attached this node
        # interrupts the run: the state persists at the gate (thread_id
        # = the shipment record id) until the service resumes the
        # thread with the human's decision (see resume_approval). The
        # node itself records the decision in the graph state — it
        # sends nothing, files nothing; the service layer owns the
        # decision record and any dispatch. Without a checkpointer the
        # node is a pass-through and the run simply ends, as before.
        # (Not event-wrapped: its "duration" would span the human wait.)
        if checkpointer is None:
            return {}
        decision = interrupt(
            {
                "gate": "human_approval",
                "shipment_id": state["shipment"]["shipment_id"],
                "status": "awaiting_approval",
            }
        )
        if isinstance(decision, dict) and decision.get("decision") in (
            "approved",
            "rejected",
        ):
            return {"approval_status": decision["decision"]}
        return {}

    # --- Node wrappers: run events + durations -------------------------
    # Every node is wrapped once, here, so event emission and duration
    # measurement live in exactly one place instead of eleven. The
    # wrapper also derives the semantic events from the node's state
    # update (the diagnosis' tool calls, the guardrail verdict, a
    # repair attempt) — the nodes themselves stay event-agnostic.
    durations: dict[str, float] = {}
    if run_context is not None:
        run_context["durations"] = durations

    def _emit_semantic(name: str, update: dict, shipment_id: str) -> None:
        if event_sink is None:
            return
        if name == "diagnose":
            for call in (update.get("diagnosis") or {}).get("tool_calls", []):
                event_sink.emit(
                    RunEvent(
                        type="tool_called",
                        shipment_id=shipment_id,
                        node=name,
                        detail={"tool": call["name"], "summary": call["summary"]},
                    )
                )
        elif name == "validate":
            validation = update.get("validation") or {}
            event_sink.emit(
                RunEvent(
                    type="guardrail_verdict",
                    shipment_id=shipment_id,
                    node=name,
                    detail={
                        "passed": bool(validation.get("passed")),
                        "errors": list(validation.get("errors", [])),
                    },
                )
            )
            if update.get("repair_attempted"):
                event_sink.emit(
                    RunEvent(
                        type="repair_attempted",
                        shipment_id=shipment_id,
                        node=name,
                        detail={
                            "attempts": int(update.get("repair_attempts") or 0),
                            "repaired": bool(update.get("repaired")),
                        },
                    )
                )

    def _wrap(name, fn):
        def wrapped(state: AgentState) -> AgentState:
            shipment_id = (state.get("shipment") or {}).get("shipment_id", "")
            if event_sink is not None:
                event_sink.emit(
                    RunEvent(type="node_started", shipment_id=shipment_id, node=name)
                )
            started = time.perf_counter()
            with node_span(name, shipment_id):
                update = fn(state)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            durations[name] = elapsed_ms
            if event_sink is not None:
                event_sink.emit(
                    RunEvent(
                        type="node_finished",
                        shipment_id=shipment_id,
                        node=name,
                        duration_ms=elapsed_ms,
                    )
                )
                _emit_semantic(name, update, shipment_id)
            return update

        return wrapped

    graph = StateGraph(AgentState)
    graph.add_node("extract", _wrap("extract", extract))
    graph.add_node("ingest", _wrap("ingest", ingest))
    graph.add_node("classify", _wrap("classify", classify))
    graph.add_node("retrieve", _wrap("retrieve", retrieve))
    graph.add_node("discrepancies", _wrap("discrepancies", discrepancies))
    graph.add_node("history", _wrap("history", history))
    graph.add_node("diagnose", _wrap("diagnose", diagnose))
    graph.add_node("options", _wrap("options", options))
    graph.add_node("draft", _wrap("draft", draft))
    graph.add_node("verify", _wrap("verify", verify))
    graph.add_node("review", _wrap("review", review))
    graph.add_node("validate", _wrap("validate", validate))
    graph.add_node("human_approval", _wrap("human_approval", human_approval))
    graph.add_node("approval_gate", approval_gate)
    if evidence_mode == "parallel":
        # Fan-out: extract and ingest are independent; classification
        # joins them (its LLM half reads the computed facts). After
        # classification, retrieval, the extraction cross-check, and
        # the history evidence are mutually independent — they run
        # concurrently and merge (disjoint state keys, so the merge
        # is order-stable) before diagnose. LangGraph branches were
        # chosen over a thread pool inside one node because each
        # branch stays a named, traced, individually timed node.
        graph.add_edge(START, "extract")
        graph.add_edge(START, "ingest")
        graph.add_edge("extract", "classify")
        graph.add_edge("ingest", "classify")
        for branch in ("retrieve", "discrepancies", "history"):
            graph.add_edge("classify", branch)
            graph.add_edge(branch, "diagnose")
    else:
        graph.add_edge(START, "extract")
        for source, target in [
            ("extract", "ingest"),
            ("ingest", "classify"),
            ("classify", "retrieve"),
            ("retrieve", "discrepancies"),
            ("discrepancies", "history"),
            ("history", "diagnose"),
        ]:
            graph.add_edge(source, target)
    for source, target in [
        ("diagnose", "options"),
        ("options", "draft"),
        ("draft", "verify"),
        ("verify", "review"),
        ("review", "validate"),
        ("validate", "human_approval"),
        ("human_approval", "approval_gate"),
        ("approval_gate", END),
    ]:
        graph.add_edge(source, target)
    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def run_shipment(
    shipment: ShipmentInput | dict,
    backend: ModelBackend | None = None,
    retriever: Retriever | None = None,
    history: dict | None = None,
    priors: list[dict] | None = None,
    event_sink: EventSink | None = None,
    checkpointer: Checkpointer | None = None,
    thread_id: str | None = None,
    evidence_mode: str = "parallel",
) -> AgentResult:
    """Run one shipment through the full graph and return the typed result.

    ``history`` is the memory summary for this shipment's consignee and
    lane (see ``service.ShipmentService.analyze``); it reaches the
    diagnosis as evidence. ``priors`` is the raw stored-history entries
    behind that summary — the diagnosis tool loop reads them (lane and
    carrier history tools). Direct callers usually leave both None.

    ``event_sink`` receives the run's structured events (see
    ``events.py``): run_started, per-node start/finish with durations,
    tool calls, the guardrail verdict, repair attempts, and
    run_completed / run_failed. Leave it None for a silent run.

    ``checkpointer`` + ``thread_id`` (default: the shipment id) run
    the graph checkpointed: it pauses at the approval gate with its
    state persisted, ready for :func:`resume_approval`. Each analysis
    starts a FRESH thread — any earlier thread for the same record is
    deleted first — because re-analysing a shipment supersedes the
    earlier run (the store replaces the record the same way).
    """
    shipment_model = (
        shipment if isinstance(shipment, ShipmentInput) else ShipmentInput.model_validate(shipment)
    )
    effective_backend = backend or MockModelBackend()
    run_context: dict = {}
    app = build_graph(
        backend=effective_backend,
        retriever=retriever,
        event_sink=event_sink,
        run_context=run_context,
        checkpointer=checkpointer,
        evidence_mode=evidence_mode,
    )
    invoke_config = None
    if checkpointer is not None:
        resolved_thread = thread_id or shipment_model.shipment_id
        checkpointer.delete_thread(resolved_thread)
        invoke_config = {"configurable": {"thread_id": resolved_thread}}
    # Telemetry: snapshot the backend's cumulative usage around the run
    # (a service reuses one backend across runs, so the run's share is
    # the delta), and time the whole invoke on the wall clock.
    usage_totals = getattr(effective_backend, "usage_totals", None)
    before = usage_totals() if callable(usage_totals) else None
    started = time.perf_counter()
    if event_sink is not None:
        event_sink.emit(RunEvent(type="run_started", shipment_id=shipment_model.shipment_id))
    try:
        with run_span(shipment_model.shipment_id):
            final = app.invoke(
                {
                    "shipment": shipment_model.model_dump(mode="json"),
                    "history": history,
                    "priors": priors or [],
                },
                invoke_config,
            )
    except Exception as exc:
        if event_sink is not None:
            event_sink.emit(
                RunEvent(
                    type="run_failed",
                    shipment_id=shipment_model.shipment_id,
                    detail={"error": str(exc)},
                )
            )
        raise
    latency = round(time.perf_counter() - started, 3)
    if event_sink is not None:
        event_sink.emit(
            RunEvent(
                type="run_completed",
                shipment_id=shipment_model.shipment_id,
                detail={"status": final.get("approval_status", "awaiting_approval")},
            )
        )
    telemetry = _run_telemetry(
        effective_backend, before, latency, budget_limit=_token_budget()
    )
    result = AgentResult(
        shipment_id=shipment_model.shipment_id,
        classification=final["classification"],
        llm_suggestion=final.get("classification_suggestion"),
        cross_check=final.get("cross_check"),
        extractions=final.get("extractions", []),
        injection_flags=final.get("injection_flags", []),
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
        trace=_build_trace(
            shipment_model,
            final,
            durations=run_context.get("durations"),
            retry_notes=run_context.get("retry_log"),
        ),
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


def resume_approval(
    thread_id: str,
    decision: dict,
    checkpointer: Checkpointer | None,
) -> dict | None:
    """Resume a checkpointed thread paused at the approval gate.

    ``decision`` is the resume payload the gate node records, e.g.
    ``{"decision": "approved", "actor": "…", "reason": "…"}``. Returns
    the final graph state when the thread was pending and has now
    completed, or ``None`` when there is nothing to resume — no
    checkpointer, an unknown thread, or a thread that already
    completed (e.g. the analysis ran with checkpoints off, or the
    decision was already applied). Callers treat the store as the
    record of the decision either way; this is the graph-state half.

    The graph is rebuilt with the deterministic fallback backend on
    purpose: the only node left to run is the gate, which does no
    model work, so a resume never touches a provider.
    """
    if checkpointer is None:
        return None
    app = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = app.get_state(config)
    except Exception:
        return None
    if not snapshot.next:  # nothing pending on this thread
        return None
    return app.invoke(Command(resume=decision), config)


def _run_telemetry(
    backend, before: dict | None, latency: float, budget_limit: int | None = None
) -> RunTelemetry:
    """Aggregate one run's telemetry from the backend's usage counters.

    Provider mode: real token usage (the delta over the run), the model
    that served it, and the estimated cost from the price table (None
    for an unlisted model). Mock mode: tokens and cost are None — no
    model ran, and the telemetry says so instead of inventing numbers —
    while the call count and latency remain real. When
    ``RUN_TOKEN_BUDGET`` is set, the budget accounting (limit, tokens
    used, whether it was exceeded) rides along.
    """
    usage_totals = getattr(backend, "usage_totals", None)
    after = usage_totals() if callable(usage_totals) else None
    backend_name = getattr(backend, "name", "mock")
    if before is None or after is None:
        return RunTelemetry(backend=backend_name, latency_seconds=latency)
    calls = max(int(after.get("calls", 0)) - int(before.get("calls", 0)), 0)
    input_tokens = max(
        int(after.get("input_tokens", 0)) - int(before.get("input_tokens", 0)), 0
    )
    output_tokens = max(
        int(after.get("output_tokens", 0)) - int(before.get("output_tokens", 0)), 0
    )
    budget = None
    if budget_limit:
        used = input_tokens + output_tokens
        budget = TokenBudget(
            limit=budget_limit, used=used, exceeded=used > budget_limit
        )
    if backend_name == "mock":
        return RunTelemetry(
            backend=backend_name,
            model_calls=calls,
            latency_seconds=latency,
            budget=budget,
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
        budget=budget,
    )
