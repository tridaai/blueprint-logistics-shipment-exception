"""CLI — batch-style runner over sample shipments (for the traced
single-shipment walkthrough, use the demo: ``python -m shipment_agent demo``).

Usage:
    shipment-agent                       # first bundled sample shipment
    shipment-agent --all                 # every bundled sample shipment
    shipment-agent --index 2             # a specific bundled sample
    shipment-agent --file data/sample/sample_shipments.json --all
    shipment-agent --all --concurrency 4 # the batch, four at a time

The default data source is the sample set bundled inside the package, so
this works identically from a source checkout and a pip install.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import silence_langchain_deprecation_warnings

# Must precede the graph import: the warning fires while langgraph loads.
silence_langchain_deprecation_warnings()

from .graph import run_shipment
from .ports import ModelBackend, Retriever
from .samples import load_sample_shipments
from .schemas import AgentResult, ShipmentInput
from .service import BatchItem
from .store import InMemoryStore
from .wiring import build_backend, build_retriever, build_service_from_env


def _load_shipments(path: Path | None) -> list[ShipmentInput]:
    if path is None:
        return [ShipmentInput.model_validate(item) for item in load_sample_shipments()]
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ShipmentInput.model_validate(item) for item in raw]


def _print_result(
    shipment: ShipmentInput,
    backend: ModelBackend | None = None,
    retriever: Retriever | None = None,
) -> None:
    result = run_shipment(shipment, backend=backend, retriever=retriever)
    _render_result(shipment, result)


def _render_result(shipment: ShipmentInput, result: AgentResult) -> None:
    c = result.classification
    print("=" * 72)
    print(f"Shipment {result.shipment_id}: {shipment.origin} -> {shipment.destination}")
    print(f"Exception : {c.exception_type.value} (severity {c.severity.value}, confidence {c.confidence})")
    print(f"Rationale : {c.rationale}")
    if result.cross_check is not None:
        cc = result.cross_check
        llm_part = (
            f"llm={cc.llm_exception_type} ({cc.llm_confidence})"
            if cc.llm_exception_type is not None
            else "llm=unavailable"
        )
        print(f"Cross-check: rules={cc.rule_exception_type} ({cc.rule_confidence}) vs {llm_part} -> {cc.resolution}")
    if result.delay_hours is not None:
        print(f"Delay     : {result.delay_hours} hours vs schedule")
    if result.document_mismatches:
        print(f"Mismatches: {[m.model_dump() for m in result.document_mismatches]}")
    print(f"Policies  : {[p.policy_id for p in result.policies]}")
    if result.diagnosis is not None:
        print(f"Diagnosis : {result.diagnosis.root_cause} (composed by: {result.diagnosis.source})")
    recommended = next((o for o in result.recovery_options if o.recommended), None)
    if recommended is not None:
        print(f"Recovery  : recommended {recommended.option_id} [{recommended.kind}] {recommended.title} (score {recommended.score}) of {len(result.recovery_options)} scored option(s)")
    if result.verification is not None:
        verdict = "grounded" if result.verification.grounded else "NOT GROUNDED"
        issues = f" issues={result.verification.issues}" if result.verification.issues else ""
        print(f"Self-check: {verdict} ({result.verification.source}){issues}")
    if result.review is not None:
        findings = f" findings={result.review.findings}" if result.review.findings else ""
        blocked = " — REVIEWER BLOCKED (flagged for the approver)" if result.reviewer_blocked else ""
        print(f"Review    : {result.review.verdict.upper()} ({result.review.source}){findings}{blocked}")
    print(f"Guardrails: passed={result.validation.passed} errors={result.validation.errors}")
    if result.repair_attempted:
        print(f"Repair    : attempted ({result.repair_attempts}) — {'repaired' if result.repaired else 'still failing'}; original errors={result.original_validation.errors if result.original_validation else []}")
    print(f"Approval  : {result.approval_status} | external action taken: {result.external_action_taken}")
    if result.autonomy is not None:
        label = "eligible for auto-approval" if result.autonomy.eligible_for_auto_approval else "human decision required"
        print(f"Autonomy  : {label} (recommendation only) — {'; '.join(result.autonomy.reasons)}")
    if result.needs_information and result.information_request is not None:
        print(f"Info needed: clarification request composed (not sent) — missing: {'; '.join(result.information_request.missing_items)}")
    if result.telemetry is not None:
        t = result.telemetry
        budget_text = ""
        if t.budget is not None:
            budget_text = f" · token budget {t.budget.used}/{t.budget.limit}" + (
                " EXCEEDED (later steps degraded)" if t.budget.exceeded else ""
            )
        if t.input_tokens is not None:
            cost = f" · est. ${t.estimated_cost_usd:.4f}" if t.estimated_cost_usd is not None else " · cost n/a"
            print(f"Telemetry : {t.backend} · {t.model} · {t.model_calls} model call(s) · {t.input_tokens} in / {t.output_tokens} out tokens{cost} · {t.latency_seconds}s{budget_text}")
        else:
            print(f"Telemetry : {t.backend} (offline fallback — no model ran) · {t.model_calls} call(s) · tokens n/a · {t.latency_seconds}s{budget_text}")
    print("-" * 72)
    print(f"Subject: {result.draft.subject}")
    print(result.draft.body)
    print()


def _run_batch(shipments: list[ShipmentInput], concurrency: int) -> list[BatchItem]:
    """The concurrent batch path: the service layer with an in-memory
    store (the CLI persists nothing), each item rendered as it lands.
    Configuration was already resolved by main() before this runs, so a
    misconfigured backend fails loudly there, not per item here."""
    service = build_service_from_env(store=InMemoryStore())
    items = service.analyze_batch(shipments, concurrency=concurrency)
    for shipment, item in zip(shipments, items):
        if item.error:
            print("=" * 72)
            print(f"Shipment {item.shipment_id}: ERROR — {item.error}")
            print()
        elif item.result is not None:
            _render_result(shipment, item.result)
    return items


def _print_batch_summary(items: list[BatchItem], concurrency: int, wall_clock: float) -> None:
    print("=" * 72)
    print(
        f"Batch summary: {len(items)} shipment(s) · concurrency {concurrency} "
        f"· wall-clock {wall_clock:.2f}s"
    )
    for item in items:
        if item.error:
            print(f"  {item.shipment_id}: ERROR — {item.error}")
        elif item.result is not None:
            result = item.result
            print(
                f"  {item.shipment_id}: {result.classification.exception_type.value} "
                f"— {result.approval_status}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Shipment Exception Agent CLI (offline, synthetic data)")
    parser.add_argument("--file", type=Path, default=None, help="Path to a shipments JSON file (default: bundled samples)")
    parser.add_argument("--index", type=int, default=0, help="Which sample shipment to run")
    parser.add_argument("--all", action="store_true", help="Run every sample shipment")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="With --all: analyse N shipments concurrently (default 1 = sequential, unchanged)",
    )
    args = parser.parse_args()

    # Backend + retriever come from the environment (.env loaded by the
    # factories). A misconfiguration (e.g. MODEL_BACKEND=openai with no
    # key) fails loudly with the actionable message, not a traceback —
    # and so does a provider failure mid-run (e.g. Ollama not running):
    # provider errors arrive pre-translated via errors.ProviderError.
    try:
        backend = build_backend()
        retriever = build_retriever()
        shipments = _load_shipments(args.file)
        selected = shipments if args.all else [shipments[args.index]]
        started = time.perf_counter()
        if args.all and args.concurrency > 1:
            items = _run_batch(selected, args.concurrency)
        else:
            items = []
            for shipment in selected:
                result = run_shipment(shipment, backend=backend, retriever=retriever)
                _render_result(shipment, result)
                items.append(
                    BatchItem(shipment_id=shipment.shipment_id, result=result)
                )
        if args.all:
            _print_batch_summary(items, args.concurrency, time.perf_counter() - started)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
