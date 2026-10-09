"""Golden-dataset evaluation for the Shipment Exception Agent.

Runs the deterministic classifier (the same code the agent graph uses) over
``golden.jsonl`` and reports exception-type accuracy and severity accuracy.
All cases are synthetic. Run from the repo root:

    python evals/run_evals.py

Exit code is 0 when type accuracy >= 90%, 1 otherwise (the gate).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shipment_agent.classifier import classify_shipment  # noqa: E402
from shipment_agent.schemas import ShipmentInput  # noqa: E402
from shipment_agent.tools import compare_documents  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
THRESHOLD = 0.90


def _write_results(payload: dict) -> None:
    """Emit results.json next to this script (and into the package, when the
    source tree is present) so the demo console can display the summary."""
    targets = [Path(__file__).resolve().parent / "results.json"]
    packaged = (
        Path(__file__).resolve().parents[1]
        / "src" / "shipment_agent" / "data" / "eval_results.json"
    )
    if packaged.parent.is_dir():
        targets.append(packaged)
    for target in targets:
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    cases = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    type_correct = 0
    severity_correct = 0
    per_type_total: dict[str, int] = defaultdict(int)
    per_type_correct: dict[str, int] = defaultdict(int)
    failures: list[str] = []

    for case in cases:
        shipment = ShipmentInput.model_validate(case["shipment"])
        result = classify_shipment(shipment, compare_documents(shipment.documents))
        expected_type = case["expected_type"]
        per_type_total[expected_type] += 1
        if result.exception_type.value == expected_type:
            type_correct += 1
            per_type_correct[expected_type] += 1
        else:
            failures.append(
                f"{case['case_id']}: expected {expected_type}, got {result.exception_type.value}"
            )
        if result.severity.value == case["expected_severity"]:
            severity_correct += 1

    total = len(cases)
    type_accuracy = type_correct / total
    severity_accuracy = severity_correct / total

    from datetime import datetime, timezone

    _write_results(
        {
            "dataset": "golden.jsonl (synthetic)",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_cases": total,
            "type_correct": type_correct,
            "type_accuracy": round(type_accuracy, 4),
            "severity_correct": severity_correct,
            "severity_accuracy": round(severity_accuracy, 4),
            "per_type": {
                etype: {
                    "correct": per_type_correct[etype],
                    "total": per_type_total[etype],
                    "accuracy": round(per_type_correct[etype] / per_type_total[etype], 4),
                }
                for etype in sorted(per_type_total)
            },
            "threshold": THRESHOLD,
            "passed": type_accuracy >= THRESHOLD,
            "framing": (
                "Synthetic golden set used as a regression gate. "
                "It is not a real-world accuracy claim."
            ),
        }
    )

    print(f"Golden dataset: {total} synthetic cases")
    print(f"Exception-type accuracy : {type_correct}/{total} = {type_accuracy:.1%}")
    print(f"Severity accuracy       : {severity_correct}/{total} = {severity_accuracy:.1%}")
    for etype in sorted(per_type_total):
        print(f"  {etype:<20} {per_type_correct[etype]}/{per_type_total[etype]}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
    print("PASS" if type_accuracy >= THRESHOLD else f"FAIL (threshold {THRESHOLD:.0%})")
    return 0 if type_accuracy >= THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
