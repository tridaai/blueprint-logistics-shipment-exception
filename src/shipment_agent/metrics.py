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

The same exposition also carries the **worker families** when the
store holds worker status rows: the background processes (the
dispatch-retry worker, the SLA sweep) record a run summary per
sweep, and :func:`worker_metrics` projects those rows — last sweep
time, sweep counts, cumulative outcomes per tenant — so a worker
that stopped sweeping is visible on the same scrape as the record
aggregates instead of failing silently in another process.
"""

from __future__ import annotations

from datetime import datetime

from .store import ApprovalRecord


def worker_metrics(status_by_worker: dict[str, dict]) -> dict:
    """Project the workers' status rows into metric shape.

    The background processes (the dispatch-retry worker, the SLA
    sweep) record a summary row per sweep in the store (see
    ``service._record_worker_status``); this turns the rows into
    what /metrics renders: when each worker last swept, how many
    sweeps it has run, and its cumulative outcome counts per tenant.
    Rows are operator data, not record data — the projection is a
    pass over summaries, never over shipments.
    """
    workers: dict[str, dict] = {}
    for name in sorted(status_by_worker):
        summary = status_by_worker[name] or {}
        last = summary.get("last_sweep_at")
        timestamp = None
        if last:
            try:
                timestamp = datetime.fromisoformat(
                    str(last).replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                timestamp = None
        workers[name] = {
            "last_sweep_timestamp": timestamp,
            "sweeps_total": int(summary.get("sweeps", 0)),
            "by_tenant": summary.get("by_tenant") or {},
        }
    return workers


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


def render_prometheus(metrics: dict, worker_status: dict | None = None) -> str:
    """The metrics payload as Prometheus text exposition.

    One HELP/TYPE header per metric family, then one sample line per
    label set — the shape strict parsers expect. ``worker_status``
    (the store's worker rows, keyed by worker name) adds the worker
    families when given: the background processes' heartbeat beside
    the record aggregates.
    """
    lines: list[str] = []

    def family(name: str, help_text: str, samples: list[tuple[dict | None, object]]):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        for labels, value in samples:
            label_text = ""
            if labels:
                label_text = "{" + ",".join(
                    f'{k}="{_escape_label(str(v))}"' for k, v in labels.items()
                ) + "}"
            lines.append(f"{name}{label_text} {value}")

    family(
        "shipment_agent_runs_total",
        "Analyses recorded in the store.",
        [(None, metrics["runs_total"])],
    )
    family(
        "shipment_agent_decisions_total",
        "Human decisions recorded, by outcome.",
        [
            ({"decision": decision}, count)
            for decision, count in metrics["decisions"].items()
        ],
    )
    family(
        "shipment_agent_guardrail_failed_runs_total",
        "Runs whose final draft failed guardrail validation.",
        [(None, metrics["guardrail_failed_runs"])],
    )
    family(
        "shipment_agent_guardrail_check_failures_total",
        "Failed guardrail checks, by check name.",
        [
            ({"check": check}, count)
            for check, count in sorted(metrics["guardrail_check_failures"].items())
        ],
    )
    family(
        "shipment_agent_latency_seconds_total",
        "Summed per-run wall-clock latency (provider runs).",
        [(None, metrics["latency_seconds_total"])],
    )
    family(
        "shipment_agent_latency_seconds_avg",
        "Mean per-run wall-clock latency across recorded runs.",
        [(None, metrics["latency_seconds_avg"])],
    )
    family(
        "shipment_agent_model_calls_total",
        "Summed model calls over runs that report telemetry.",
        [(None, metrics["model_calls_total"])],
    )
    family(
        "shipment_agent_input_tokens_total",
        "Summed provider-reported input tokens (runs that report them).",
        [(None, metrics["input_tokens_total"])],
    )
    family(
        "shipment_agent_output_tokens_total",
        "Summed provider-reported output tokens (runs that report them).",
        [(None, metrics["output_tokens_total"])],
    )
    family(
        "shipment_agent_estimated_cost_usd_total",
        "Summed estimated provider cost over runs that report it.",
        [(None, metrics["estimated_cost_usd_total"])],
    )
    if worker_status:
        workers = worker_metrics(worker_status)
        family(
            "shipment_agent_worker_last_sweep_timestamp",
            "When each background worker last completed a sweep (unix seconds). "
            "A stale value is a worker that has stopped sweeping.",
            [
                ({"worker": name}, round(w["last_sweep_timestamp"], 3))
                for name, w in workers.items()
                if w["last_sweep_timestamp"] is not None
            ],
        )
        family(
            "shipment_agent_worker_sweeps_total",
            "Sweeps completed by each background worker, recorded in the store.",
            [({"worker": name}, w["sweeps_total"]) for name, w in workers.items()],
        )
        outcome_samples: list[tuple[dict | None, object]] = []
        for name, w in workers.items():
            for tenant in sorted(w["by_tenant"]):
                for outcome in sorted(w["by_tenant"][tenant]):
                    outcome_samples.append(
                        (
                            {
                                "worker": name,
                                "tenant": tenant,
                                "outcome": outcome,
                            },
                            w["by_tenant"][tenant][outcome],
                        )
                    )
        family(
            "shipment_agent_worker_outcomes_total",
            "Cumulative outcomes recorded by each background worker, per tenant.",
            outcome_samples,
        )
    return "\n".join(lines) + "\n"
