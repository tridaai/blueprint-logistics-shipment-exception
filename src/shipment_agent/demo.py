"""One-command demo: ONE shipment, end to end, with a readable trace.

    shipment-agent-demo            (console script)
    python -m shipment_agent demo  (module form)
    make demo

Default: offline, deterministic mock backend, synthetic sample data. Set
MODEL_BACKEND=openai|anthropic (plus the API key, e.g. in .env) and the
same trace runs against the real provider — the header line says which
backend produced the draft. The trace shows every stage a reviewer needs
to judge the blueprint: input → classification with evidence → retrieved
policy → draft + claim packet → guardrail checks → the human-approval
gate it stops at.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import silence_langchain_deprecation_warnings

# Must precede the graph import: the warning fires while langgraph loads.
silence_langchain_deprecation_warnings()

from .graph import run_shipment
from .model_backends import ModelBackend, get_backend
from .retriever import Retriever, get_retriever
from .samples import sample_shipment_models

_LINE = "=" * 74


def print_trace(
    index: int = 0,
    backend: ModelBackend | None = None,
    retriever: Retriever | None = None,
) -> None:
    backend = backend or get_backend()
    retriever = retriever or get_retriever()
    shipments = sample_shipment_models()
    shipment = shipments[index]
    result = run_shipment(shipment, backend=backend, retriever=retriever)
    c = result.classification
    backend_desc = (
        "offline mock backend"
        if backend.name == "mock"
        else f"{backend.name} backend via MODEL_BACKEND (real provider calls)"
    )

    print(_LINE)
    print("TRIDA AI BLUEPRINT DEMO — Logistics Shipment Exception Agent")
    print(f"Reference prototype · synthetic data · {backend_desc} · no external action")
    print(_LINE)

    print("\n[1] INPUT SHIPMENT")
    print(f"    ID          : {shipment.shipment_id}")
    print(f"    Route       : {shipment.origin} -> {shipment.destination} ({shipment.carrier})")
    print(f"    Customer    : {shipment.customer_name} · service level: {shipment.service_level}")
    print(f"    Status      : {shipment.status}")
    print(f"    Latest event: {shipment.latest_event or '—'}")
    if shipment.condition_notes:
        print(f"    Condition   : {shipment.condition_notes}")
    print(f"    Documents   : {[d.document_id for d in shipment.documents] or 'none'}")
    if result.delay_hours is not None:
        print(f"    Computed    : delay vs schedule = {result.delay_hours} hours")
    if result.document_mismatches:
        for m in result.document_mismatches:
            print(f"    Computed    : doc mismatch on '{m.field}': BOL={m.bol_value} vs invoice={m.invoice_value}")

    print("\n[2] CLASSIFICATION")
    print(f"    Exception : {c.exception_type.value}")
    print(f"    Severity  : {c.severity.value}   Confidence: {c.confidence}")
    print(f"    Rationale : {c.rationale}")
    print("    Evidence (signals):")
    for signal in c.signals or ["(none)"]:
        print(f"      - {signal}")
    if result.cross_check is not None:
        cc = result.cross_check
        print(f"    Rules       : {cc.rule_exception_type} · severity {cc.rule_severity} · confidence {cc.rule_confidence}")
        if cc.llm_exception_type is not None:
            print(f"    LLM check   : {cc.llm_exception_type} · severity {cc.llm_severity} · confidence {cc.llm_confidence} (via {cc.llm_backend})")
        else:
            print("    LLM check   : no usable LLM classification — rules only")
        print(f"    Cross-check : {cc.resolution} — {cc.note}")

    print("\n[3] RETRIEVED POLICY CONTEXT")
    for policy in result.policies:
        snippet = policy.snippet if len(policy.snippet) <= 160 else policy.snippet[:157] + "..."
        print(f"    [{policy.policy_id}] {policy.title} (score {policy.score})")
        print(f"        {snippet}")
    if not result.policies:
        print("    (no policy retrieved)")

    print("\n[4] DIAGNOSIS")
    if result.diagnosis is not None:
        print(f"    Root cause: {result.diagnosis.root_cause}")
        print(f"    Summary   : {result.diagnosis.summary} (composed by: {result.diagnosis.source})")
        for line in result.diagnosis.evidence:
            print(f"      - evidence: {line}")

    print("\n[5] RECOVERY OPTIONS (scored by code — the model never does this arithmetic)")
    for option in result.recovery_options:
        mark = " <- recommended" if option.recommended else ""
        print(
            f"    {option.option_id} [{option.kind}] {option.title} — score {option.score} "
            f"(ETA +{option.eta_improvement_hours}h · added cost {option.added_cost_units} units · SLA {option.sla_score}){mark}"
        )

    print("\n[6] DRAFT — CUSTOMER UPDATE")
    print(f"    Subject: {result.draft.subject}")
    for line in result.draft.body.splitlines():
        print(f"    {line}")

    print("\n[7] DRAFT — CLAIM PACKET (not filed)")
    print("    " + json.dumps(result.draft.claim_packet, indent=2, ensure_ascii=False).replace("\n", "\n    "))

    print("\n[8] SELF-VERIFICATION — the agent critiques its own draft")
    if result.verification is not None:
        verdict = "GROUNDED" if result.verification.grounded else "NOT GROUNDED"
        source = "LLM critique" if result.verification.source == "llm" else "deterministic checklist"
        print(f"    verdict: {verdict} ({source})")
        if result.verification.summary:
            print(f"    {result.verification.summary}")
        for issue in result.verification.issues:
            print(f"    issue: {issue}")
        if result.verification.note:
            print(f"    note: {result.verification.note}")

    print("\n[9] INDEPENDENT REVIEW — a second pair of eyes (not the drafter)")
    if result.review is not None:
        r = result.review
        source = (
            f"LLM reviewer{(' · ' + r.model) if r.model else ''}"
            if r.source == "llm"
            else "deterministic checklist"
        )
        print(f"    verdict: {r.verdict.upper()} ({source})")
        for finding in r.findings:
            print(f"    finding: {finding}")
        if r.note:
            print(f"    note: {r.note}")
        if result.reviewer_blocked:
            print("    ⛔ REVIEWER BLOCKED this draft — flagged for the approver; the human still decides.")
    else:
        print("    (disabled — REVIEWER=off)")

    print("\n[10] GUARDRAIL CHECKS")
    for check in result.validation.checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"    [{mark}] {check.name} — {check.detail}")
    for warning in result.validation.warnings:
        print(f"    [warn] {warning}")
    print(f"    Overall: {'PASSED' if result.validation.passed else 'FAILED'}")
    if result.repair_attempted:
        outcome = "redraft passed" if result.repaired else "redraft still failed"
        print(f"    Repair: bounded repair attempted ({result.repair_attempts} attempt(s)) — {outcome}")
        if result.original_validation is not None:
            for error in result.original_validation.errors:
                print(f"    Original failure: {error}")

    print("\n[11] FINAL STATE")
    print("    PENDING_HUMAN_APPROVAL" if result.approval_status == "awaiting_approval" else f"    {result.approval_status}")
    print(f"    approval_status = {result.approval_status} · external_action_taken = {result.external_action_taken}")
    if result.autonomy is not None:
        label = "eligible for auto-approval" if result.autonomy.eligible_for_auto_approval else "human decision required"
        print(f"    Autonomy recommendation: {label} (recommendation only — never acted on)")
        for reason in result.autonomy.reasons:
            print(f"      - {reason}")
    if result.needs_information and result.information_request is not None:
        print("    Information : NEEDED — this case is under-determined; a clarification")
        print("                  request was composed for the carrier/ops contact (not sent):")
        for item in result.information_request.missing_items:
            print(f"      - missing: {item}")
    if result.telemetry is not None:
        t = result.telemetry
        if t.input_tokens is not None:
            cost = (
                f" · est. ${t.estimated_cost_usd:.4f}"
                if t.estimated_cost_usd is not None
                else " · cost n/a (model not in the price table)"
            )
            print(
                f"    Telemetry : {t.backend} · {t.model} · {t.model_calls} model call(s) · "
                f"{t.input_tokens} in / {t.output_tokens} out tokens{cost} · {t.latency_seconds}s"
            )
        else:
            print(
                f"    Telemetry : {t.backend} (offline fallback — no model ran) · "
                f"{t.model_calls} call(s) · tokens n/a · {t.latency_seconds}s"
            )
    print("    Nothing was sent or filed. A human approves via the API/UI before any action.")
    print(_LINE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shipment Exception Agent — traced demo (offline)")
    parser.add_argument("--index", type=int, default=0, help="Which bundled sample shipment to run (0-based)")
    args = parser.parse_args()
    try:
        print_trace(args.index)
    except (RuntimeError, ValueError) as exc:
        # e.g. MODEL_BACKEND=openai with no key — fail loudly and cleanly.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
