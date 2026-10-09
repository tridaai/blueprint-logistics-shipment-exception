"""LLM-mode evaluation pack — opt-in, needs a real provider key.

This is NOT the default gate (``run_evals.py`` is, and it stays
deterministic and offline). This pack answers the provider-mode
questions the golden gate cannot:

1. **Cross-check agreement** — how often do the deterministic rules and
   the LLM classify the golden cases the same way, and how do the
   disagreements resolve (``agree`` / ``rules_authoritative`` /
   ``llm_adopted``)?
2. **Draft groundedness** — an LLM judge scores each produced draft
   against the verified facts it was given: any claim not in the
   facts, any invented ETA, any prohibited promise?

Per-case token usage and latency are printed, with a summary at the
end; estimated cost uses the small price table below (USD per 1M
tokens, input/output — check your provider's current pricing; models
not in the table report cost as n/a, and a local Ollama run is $0).

Run from the repo root, with a provider configured (see .env.example):

    MODEL_BACKEND=anthropic RETRIEVER=keyword python evals/run_llm_evals.py
    python evals/run_llm_evals.py --limit 8 --types delay,damage
    python evals/run_llm_evals.py --judge-model gpt-4o-mini

Exit codes: 2 = no real provider configured (fails loudly, by design);
1 = at least one draft judged ungrounded / invented-ETA / promise, or a
judge error; 0 = all judged drafts clean.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shipment_agent.config import env_str  # noqa: E402
from shipment_agent.graph import run_shipment  # noqa: E402
from shipment_agent.model_backends import get_backend, parse_json_object  # noqa: E402
from shipment_agent.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE  # noqa: E402
from shipment_agent.retriever import get_retriever  # noqa: E402
from shipment_agent.schemas import ShipmentInput  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
RESULTS = Path(__file__).resolve().parent / "llm_results.json"

# USD per 1M tokens (input, output). Indicative list prices — verify
# against your provider; unknown models report cost as n/a.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "llama3.1": (0.0, 0.0),
    "nomic-embed-text": (0.0, 0.0),
}


def estimate_cost(model: str, usage: dict | None) -> float | None:
    """Estimated USD cost of a usage record, or None when unpriceable."""
    if not usage or model not in PRICE_TABLE:
        return None
    input_rate, output_rate = PRICE_TABLE[model]
    return (
        usage.get("input_tokens", 0) * input_rate
        + usage.get("output_tokens", 0) * output_rate
    ) / 1_000_000


def judge_draft(backend, result, judge_model: str | None) -> dict:
    """One LLM-judge verdict for a produced draft. Never raises: a
    failed judge call is a ``judge_error`` verdict, which fails the pack."""
    recommended = next(
        (o for o in result.recovery_options if o.recommended), None
    )
    user = JUDGE_USER_TEMPLATE.format(
        exception_type=result.classification.exception_type.value,
        severity=result.classification.severity.value,
        delay_hours=result.delay_hours,
        mismatches=[m.model_dump() for m in result.document_mismatches] or "none",
        citations=", ".join(result.draft.citations) or "none",
        recommended_option=(
            f"{recommended.title} [{recommended.kind}]" if recommended else "none"
        ),
        subject=result.draft.subject,
        body=result.draft.body,
    )
    try:
        text, _usage = backend.complete_with_usage(
            JUDGE_SYSTEM_PROMPT, user, max_tokens=300, model=judge_model
        )
    except Exception as exc:  # a broken judge must not look like a pass
        return {"judge_error": f"{type(exc).__name__}: {exc}"}
    data = parse_json_object(text)
    if data is None:
        return {"judge_error": "judge reply was not a JSON object"}
    try:
        return {
            "grounded": bool(data["grounded"]),
            "invented_eta": bool(data["invented_eta"]),
            "prohibited_promise": bool(data["prohibited_promise"]),
            "score": float(data["score"]),
            "rationale": str(data.get("rationale", ""))[:300],
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {"judge_error": f"judge reply missing/invalid fields: {exc}"}


def _load_cases(limit: int | None, types: str | None) -> list[dict]:
    cases = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if types:
        wanted = {t.strip() for t in types.split(",") if t.strip()}
        cases = [c for c in cases if c["expected_type"] in wanted]
    if limit:
        cases = cases[:limit]
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM-mode eval pack (opt-in, needs a provider key)")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases")
    parser.add_argument("--types", type=str, default=None, help="Comma-separated expected types to include")
    parser.add_argument("--judge-model", type=str, default=None, help="Judge model override (default: LLM_JUDGE_MODEL or the backend's model)")
    args = parser.parse_args(argv)

    try:
        backend = get_backend()
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if backend.name == "mock" or not hasattr(backend, "complete_with_usage"):
        print(
            "error: the LLM eval pack needs a real provider — the mock backend "
            "is the offline fallback and has nothing to judge with.\n"
            "Configure one first, e.g. copy .env.example to .env and set "
            "MODEL_BACKEND=openai|anthropic (plus the API key), or "
            "MODEL_BACKEND=ollama for a local model. The deterministic gate "
            "(evals/run_evals.py) needs no key and is unchanged.",
            file=sys.stderr,
        )
        return 2
    judge_model = args.judge_model or env_str("LLM_JUDGE_MODEL")
    retriever = get_retriever()
    cases = _load_cases(args.limit, args.types)
    model_name = getattr(backend, "_model", backend.name)

    print(f"LLM eval pack — backend: {backend.name} (model: {model_name}), "
          f"judge model: {judge_model or model_name}, retriever: {getattr(retriever, 'name', 'keyword')}")
    print(f"Golden cases: {len(cases)} (synthetic)")

    resolutions: Counter[str] = Counter()
    type_matches = 0
    judged = 0
    judged_clean = 0
    failures: list[str] = []
    total_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    total_cost = 0.0
    cost_known = True
    latencies: list[float] = []
    records: list[dict] = []

    for case in cases:
        shipment = ShipmentInput.model_validate(case["shipment"])
        backend.reset_usage()
        started = time.perf_counter()
        result = run_shipment(shipment, backend=backend, retriever=retriever)
        pipeline_usage = backend.usage_totals()
        verdict = judge_draft(backend, result, judge_model)
        latency = time.perf_counter() - started
        latencies.append(latency)

        case_usage = backend.usage_totals()  # pipeline + judge
        for key in total_usage:
            total_usage[key] += case_usage[key]
        case_cost = estimate_cost(model_name, pipeline_usage)
        judge_usage = {k: case_usage[k] - pipeline_usage[k] for k in case_usage}
        judge_cost = estimate_cost(judge_model or model_name, judge_usage)
        if case_cost is None or judge_cost is None:
            cost_known = False
        else:
            total_cost += case_cost + judge_cost

        resolution = result.cross_check.resolution if result.cross_check else "rules_only"
        resolutions[resolution] += 1
        final_type = result.classification.exception_type.value
        if final_type == case["expected_type"]:
            type_matches += 1

        if "judge_error" in verdict:
            verdict_line = f"JUDGE ERROR ({verdict['judge_error']})"
            failures.append(f"{case['case_id']}: judge error: {verdict['judge_error']}")
        else:
            judged += 1
            clean = (
                verdict["grounded"]
                and not verdict["invented_eta"]
                and not verdict["prohibited_promise"]
            )
            if clean:
                judged_clean += 1
            else:
                failures.append(
                    f"{case['case_id']}: grounded={verdict['grounded']} "
                    f"invented_eta={verdict['invented_eta']} "
                    f"prohibited_promise={verdict['prohibited_promise']} — {verdict['rationale']}"
                )
            verdict_line = (
                f"judge score {verdict['score']} grounded={verdict['grounded']} "
                f"invented_eta={verdict['invented_eta']} promise={verdict['prohibited_promise']}"
            )
        tokens = case_usage["input_tokens"] + case_usage["output_tokens"]
        cost_line = (
            f"cost=${(case_cost or 0.0) + (judge_cost or 0.0):.4f}"
            if case_cost is not None and judge_cost is not None
            else "cost=n/a"
        )
        print(
            f"  {case['case_id']}: expected={case['expected_type']} final={final_type} "
            f"resolution={resolution} · {verdict_line} · tokens={tokens} · "
            f"latency={latency:.2f}s · {cost_line}"
        )
        records.append(
            {
                "case_id": case["case_id"],
                "expected_type": case["expected_type"],
                "final_type": final_type,
                "resolution": resolution,
                "verdict": verdict,
                "usage": case_usage,
                "latency_seconds": round(latency, 3),
            }
        )

    total = len(cases)
    agreement = resolutions.get("agree", 0)
    print("\nSummary")
    print(f"  Cases run            : {total}")
    print(f"  Final type = expected: {type_matches}/{total}")
    print(
        "  Cross-check          : "
        + ", ".join(f"{k}={v}" for k, v in sorted(resolutions.items()))
        + f" (rules-vs-LLM agreement {agreement}/{total})"
    )
    print(f"  Judged drafts        : {judged}, clean {judged_clean}")
    total_tokens = total_usage["input_tokens"] + total_usage["output_tokens"]
    print(f"  Tokens (total)       : {total_tokens} "
          f"(in {total_usage['input_tokens']} / out {total_usage['output_tokens']}, {total_usage['calls']} calls)")
    print(f"  Estimated cost       : ${total_cost:.4f}" if cost_known else "  Estimated cost       : n/a (model not in price table)")
    if latencies:
        print(f"  Latency              : mean {sum(latencies) / len(latencies):.2f}s, max {max(latencies):.2f}s")
    if failures:
        print("  Failures:")
        for failure in failures:
            print(f"    - {failure}")
    RESULTS.write_text(
        json.dumps(
            {
                "backend": backend.name,
                "model": model_name,
                "judge_model": judge_model or model_name,
                "cases": records,
                "summary": {
                    "total": total,
                    "type_matches": type_matches,
                    "resolutions": dict(resolutions),
                    "judged": judged,
                    "judged_clean": judged_clean,
                    "total_usage": total_usage,
                    "estimated_cost_usd": round(total_cost, 4) if cost_known else None,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    passed = not failures and judged == total
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
