"""CLI — batch-style runner over sample shipments (for the traced
single-shipment walkthrough, use the demo: ``python -m shipment_agent demo``).

Usage:
    shipment-agent                       # first bundled sample shipment
    shipment-agent --all                 # every bundled sample shipment
    shipment-agent --index 2             # a specific bundled sample
    shipment-agent --file data/sample/sample_shipments.json --all

The default data source is the sample set bundled inside the package, so
this works identically from a source checkout and a pip install.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import silence_langchain_deprecation_warnings

# Must precede the graph import: the warning fires while langgraph loads.
silence_langchain_deprecation_warnings()

from .graph import run_shipment
from .model_backends import ModelBackend, get_backend
from .retriever import Retriever, get_retriever
from .samples import load_sample_shipments
from .schemas import ShipmentInput


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
    print(f"Guardrails: passed={result.validation.passed} errors={result.validation.errors}")
    print(f"Approval  : {result.approval_status} | external action taken: {result.external_action_taken}")
    print("-" * 72)
    print(f"Subject: {result.draft.subject}")
    print(result.draft.body)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Shipment Exception Agent CLI (offline, synthetic data)")
    parser.add_argument("--file", type=Path, default=None, help="Path to a shipments JSON file (default: bundled samples)")
    parser.add_argument("--index", type=int, default=0, help="Which sample shipment to run")
    parser.add_argument("--all", action="store_true", help="Run every sample shipment")
    args = parser.parse_args()

    # Backend + retriever come from the environment (.env loaded by the
    # factories). A misconfiguration (e.g. MODEL_BACKEND=openai with no
    # key) fails loudly with the actionable message, not a traceback —
    # and so does a provider failure mid-run (e.g. Ollama not running):
    # provider errors arrive pre-translated via errors.ProviderError.
    try:
        backend = get_backend()
        retriever = get_retriever()
        shipments = _load_shipments(args.file)
        selected = shipments if args.all else [shipments[args.index]]
        for shipment in selected:
            _print_result(shipment, backend=backend, retriever=retriever)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
