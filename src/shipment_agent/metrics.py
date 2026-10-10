"""Operational metrics, computed from the approval store.

The store is the record of everything the service has done — analyses
and the human decisions on them — so fleet-level numbers (how many
runs, how they were decided, how often the guardrails stopped a
draft, what the runs cost) are a pure function of the records, not a
second bookkeeping system that can drift. :func:`compute_metrics`
aggregates; :func:`render_prometheus` renders the Prometheus text
exposition format for ``GET /metrics`` (scrape it with Prometheus,
an OTel collector, or just ``curl`` it).

Honesty rules, same as the per-run telemetry: token and cost totals
sum only the runs that report them (provider mode) and say how many
runs that was — a mock run's tokens are unknown, never zero-filled
into a fake total.
"""

from __future__ import annotations

from .store import ApprovalRecord


def compute_metrics(records: list[ApprovalRecord]) -> dict:
    """Aggregate the store's records into the metrics payload."""
    by_status: dict[str, int] = {}
    guardrail_check_failures: dict[str, int] = {}
    guardrail_failed_runs = 0
    latency_total = 0.0
    input_tokens = 0
    output_tokens = 0
    model_calls = 0
    estimated_cost = 0.0
    telemetry_runs = 0
    cost_runs = 0
    for record in records:
        result = record.result
        by_status[result.approval_status] = by_status.get(result.approval_status, 0) + 1
        if not result.validation.passed:
            guardrail_failed_runs += 1
        for check in result.validation.checks:
            if not check.passed:
                guardrail_check_failures[check.name] = (
                    guardrail_check_failures.get(check.name, 0) + 1
                )
        telemetry = result.telemetry
        if telemetry is not None:
            telemetry_runs += 1
            latency_total += telemetry.latency_seconds
            model_calls += telemetry.model_calls
            if telemetry.input_tokens is not None:
                input_tokens += telemetry.input_tokens
            if telemetry.output_tokens is not None:
                output_tokens += telemetry.output_tokens
            if telemetry.estimated_cost_usd is not None:
                estimated_cost += telemetry.estimated_cost_usd
                cost_runs += 1
    runs = len(records)
    return {
        "runs_total": runs,
        "decisions": {
            "approved": by_status.get("approved", 0),
            "rejected": by_status.get("rejected", 0),
            "awaiting": by_status.get("awaiting_approval", 0),
        },
        "guardrail_failed_runs": guardrail_failed_runs,
        "guardrail_check_failures": guardrail_check_failures,
        "latency_seconds_total": round(latency_total, 3),
        "latency_seconds_avg": round(latency_total / runs, 3) if runs else 0.0,
        "model_calls_total": model_calls,
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "estimated_cost_usd_total": round(estimated_cost, 6),
        "telemetry_runs": telemetry_runs,
        "cost_runs": cost_runs,
    }


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_prometheus(metrics: dict) -> str:
    """The metrics payload as Prometheus text exposition."""
    lines: list[str] = []

    def gauge(name: str, help_text: str, value, labels: dict[str, str] | None = None):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        label_text = ""
        if labels:
            label_text = "{" + ",".join(
                f'{k}="{_escape_label(str(v))}"' for k, v in labels.items()
            ) + "}"
        lines.append(f"{name}{label_text} {value}")

    gauge("shipment_agent_runs_total", "Analyses recorded in the store.", metrics["runs_total"])
    for decision, count in metrics["decisions"].items():
        gauge(
            "shipment_agent_decisions_total",
            "Human decisions recorded, by outcome.",
            count,
            {"decision": decision},
        )
    gauge(
        "shipment_agent_guardrail_failed_runs_total",
        "Runs whose final draft failed guardrail validation.",
        metrics["guardrail_failed_runs"],
    )
    for check, count in sorted(metrics["guardrail_check_failures"].items()):
        gauge(
            "shipment_agent_guardrail_check_failures_total",
            "Failed guardrail checks, by check name.",
            count,
            {"check": check},
        )
    gauge(
        "shipment_agent_latency_seconds_total",
        "Summed per-run wall-clock latency (provider runs).",
        metrics["latency_seconds_total"],
    )
    gauge(
        "shipment_agent_latency_seconds_avg",
        "Mean per-run wall-clock latency across recorded runs.",
        metrics["latency_seconds_avg"],
    )
    gauge(
        "shipment_agent_model_calls_total",
        "Summed model calls over runs that report telemetry.",
        metrics["model_calls_total"],
    )
    gauge(
        "shipment_agent_input_tokens_total",
        "Summed provider-reported input tokens (runs that report them).",
        metrics["input_tokens_total"],
    )
    gauge(
        "shipment_agent_output_tokens_total",
        "Summed provider-reported output tokens (runs that report them).",
        metrics["output_tokens_total"],
    )
    gauge(
        "shipment_agent_estimated_cost_usd_total",
        "Summed estimated provider cost over runs that report it.",
        metrics["estimated_cost_usd_total"],
    )
    return "\n".join(lines) + "\n"
