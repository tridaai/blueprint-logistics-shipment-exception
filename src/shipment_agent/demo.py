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
    if result.llm_suggestion is not None:
        s = result.llm_suggestion
        print(f"    LLM suggestion ({s.backend}): {s.exception_type} · severity {s.severity} · confidence {s.confidence}")
        print(f"      - {s.rationale}")
        print(
            "      - agrees with the rule result"
            if s.agrees_with_rules
            else "      - DISAGREEMENT with the rule result — the rule result stays authoritative"
        )

    print("\n[3] RETRIEVED POLICY CONTEXT")
    for policy in result.policies:
        snippet = policy.snippet if len(policy.snippet) <= 160 else policy.snippet[:157] + "..."
        print(f"    [{policy.policy_id}] {policy.title} (score {policy.score})")
        print(f"        {snippet}")
    if not result.policies:
        print("    (no policy retrieved)")

    print("\n[4] DRAFT — CUSTOMER UPDATE")
    print(f"    Subject: {result.draft.subject}")
    for line in result.draft.body.splitlines():
        print(f"    {line}")

    print("\n[5] DRAFT — CLAIM PACKET (not filed)")
    print("    " + json.dumps(result.draft.claim_packet, indent=2, ensure_ascii=False).replace("\n", "\n    "))

    print("\n[6] GUARDRAIL CHECKS")
    for check in result.validation.checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"    [{mark}] {check.name} — {check.detail}")
    for warning in result.validation.warnings:
        print(f"    [warn] {warning}")
    print(f"    Overall: {'PASSED' if result.validation.passed else 'FAILED'}")

    print("\n[7] FINAL STATE")
    print("    PENDING_HUMAN_APPROVAL" if result.approval_status == "awaiting_approval" else f"    {result.approval_status}")
    print(f"    approval_status = {result.approval_status} · external_action_taken = {result.external_action_taken}")
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
