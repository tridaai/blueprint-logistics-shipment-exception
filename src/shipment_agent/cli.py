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
from pathlib import Path

from .graph import run_shipment
from .samples import load_sample_shipments
from .schemas import ShipmentInput


def _load_shipments(path: Path | None) -> list[ShipmentInput]:
    if path is None:
        return [ShipmentInput.model_validate(item) for item in load_sample_shipments()]
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ShipmentInput.model_validate(item) for item in raw]


def _print_result(shipment: ShipmentInput) -> None:
    result = run_shipment(shipment)
    c = result.classification
    print("=" * 72)
    print(f"Shipment {result.shipment_id}: {shipment.origin} -> {shipment.destination}")
    print(f"Exception : {c.exception_type.value} (severity {c.severity.value}, confidence {c.confidence})")
    print(f"Rationale : {c.rationale}")
    if result.delay_hours is not None:
        print(f"Delay     : {result.delay_hours} hours vs schedule")
    if result.document_mismatches:
        print(f"Mismatches: {[m.model_dump() for m in result.document_mismatches]}")
    print(f"Policies  : {[p.policy_id for p in result.policies]}")
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

    shipments = _load_shipments(args.file)
    selected = shipments if args.all else [shipments[args.index]]
    for shipment in selected:
        _print_result(shipment)


if __name__ == "__main__":
    main()
