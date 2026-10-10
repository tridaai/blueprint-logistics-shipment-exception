"""Queue and carrier scorecards — computed from the store.

The approval store is the system of record; everything in this
module is a deterministic projection of it, computed in code (never
by the model):

- **Approval queue** — the shipments still awaiting a human
  decision, ordered the way an approver works them: severity first
  (critical → low), then age (oldest first). Each item carries the
  flags an approver scans for — a classification cross-check
  disagreement, a guardrail repair, a reviewer block, failing
  guardrails, an information request, auto-approval eligibility —
  so the queue itself is the daily worklist, not just a count.
  Each item also carries its **age bucket** and an **SLA view**: a
  case that waits too long is itself an exception, so every severity
  has an age budget (configurable, see :func:`sla_thresholds_from_env`)
  and an item past its budget is flagged ``sla_breach`` with the
  overrun — the queue reports its own health, not just its contents.
  Past a second threshold (the escalation ladder: a configurable
  multiple of the budget) the item's stage reads ``escalated`` —
  the state the sweep's ``sla_escalation`` event acts on.
- **Carrier scorecards** — per-carrier aggregates over the whole
  stored history: shipment count, exception mix and rate, damage
  rate, and the human decision record (approvals / rejections /
  approval rate). The scorecard for a new shipment's carrier joins
  its diagnosis evidence (see ``diagnosis.memory_evidence_lines``),
  so the approver sees the carrier's track record, not just this
  one case — memory as a decision input, not only counts.
- **Fleet baseline** — the shipment-weighted exception / damage
  rates across all carriers: the comparison point that turns a
  carrier's rates into a reliability signal for option scoring
  (see ``options.reliability_adjustment``).
- **Lane projections** — the same scorecard and baseline computed
  over one corridor (``origin -> destination``) only: a carrier can
  be fine everywhere except one lane, and the option scorer's
  reliability term reads the lane figures when the carrier has
  enough history *on that lane* (see
  ``options.reliability_adjustment``).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from .config import env_float
from .store import ApprovalRecord, format_type_counts, history_entry

_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# How long a case of each severity may await a decision before the
# wait itself becomes the exception. Defaults: a critical case gets
# 4 hours of a human's attention; a low gets four days. Deployments
# tune them per severity (QUEUE_SLA_HOURS_<SEVERITY>) — the queue is
# an operations surface, and its budgets are an operations choice.
DEFAULT_SLA_HOURS = {"critical": 4.0, "high": 24.0, "medium": 48.0, "low": 96.0}

# Age buckets, coarsest last: the buckets an approver thinks in.
AGE_BUCKETS = ("<1h", "1-4h", "4-24h", "1-3d", ">3d", "unknown")


def age_bucket(age_seconds: float | None) -> str:
    """The age bucket for a queue item (``unknown`` without a time)."""
    if age_seconds is None:
        return "unknown"
    hours = age_seconds / 3600.0
    if hours < 1:
        return "<1h"
    if hours < 4:
        return "1-4h"
    if hours < 24:
        return "4-24h"
    if hours < 72:
        return "1-3d"
    return ">3d"


def sla_thresholds_from_env() -> dict[str, float]:
    """The SLA age budgets by severity, env-tunable per severity:
    ``QUEUE_SLA_HOURS_CRITICAL`` / ``_HIGH`` / ``_MEDIUM`` / ``_LOW``
    over :data:`DEFAULT_SLA_HOURS`."""
    return {
        severity: env_float(f"QUEUE_SLA_HOURS_{severity.upper()}", default)
        for severity, default in DEFAULT_SLA_HOURS.items()
    }


# The escalation ladder's second rung: a breach that keeps aging is
# a worse problem than a fresh one, so at this multiple of the
# budget the wait escalates (see the queue item's sla_stage and the
# sweep's sla_escalation event in service.py).
DEFAULT_ESCALATION_FACTOR = 2.0


def sla_escalation_factor_from_env() -> float:
    """The escalation multiple of a severity's budget,
    ``QUEUE_SLA_ESCALATION_FACTOR`` over :data:`DEFAULT_ESCALATION_FACTOR`.

    Clamped to at least 1.0: an escalation can land at the budget,
    never before the breach it escalates."""
    return max(
        1.0, env_float("QUEUE_SLA_ESCALATION_FACTOR", DEFAULT_ESCALATION_FACTOR)
    )


def sla_escalation_factors_from_env() -> dict[str, float]:
    """The escalation multiples by severity.

    ``QUEUE_SLA_ESCALATION_FACTOR_<SEVERITY>`` overrides the global
    ``QUEUE_SLA_ESCALATION_FACTOR`` for that severity — a critical
    case should escalate faster than a low one (critical 1.5×, low
    3×, say), because the ladder's patience is a per-severity
    operations choice like the budgets themselves. Every value
    clamps to at least 1.0, the same discipline as the global
    factor: an escalation can land at the budget, never before the
    breach it escalates."""
    global_factor = sla_escalation_factor_from_env()
    return {
        severity: max(
            1.0,
            env_float(
                f"QUEUE_SLA_ESCALATION_FACTOR_{severity.upper()}", global_factor
            ),
        )
        for severity in DEFAULT_SLA_HOURS
    }


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _queue_item(
    record: ApprovalRecord,
    now: datetime,
    sla_hours: dict,
    escalation_factor: float | dict = DEFAULT_ESCALATION_FACTOR,
) -> dict:
    result = record.result
    entry = history_entry(record) or {}
    cross_check = result.cross_check
    created = _parse_iso(record.created_at)
    flags = {
        # The two classifiers disagreed (whatever the resolution) —
        # the approver is the tiebreaker's audience.
        "cross_check_disagreement": bool(
            cross_check is not None and cross_check.agrees is False
        ),
        "cross_check_resolution": cross_check.resolution if cross_check else None,
        "repair_attempted": result.repair_attempted,
        "repaired": result.repaired,
        "reviewer_blocked": result.reviewer_blocked,
        "reviewer_verdict": result.review.verdict if result.review else None,
        "guardrails_passed": result.validation.passed,
        "eligible_for_auto_approval": bool(
            result.autonomy and result.autonomy.eligible_for_auto_approval
        ),
        "needs_information": result.needs_information,
    }
    age_seconds = (
        round((now - created).total_seconds(), 1) if created else None
    )
    severity = result.classification.severity.value
    # The ladder's factor for THIS item: a per-severity map (the
    # service passes sla_escalation_factors_from_env) resolves to
    # the item's own severity; a bare float applies uniformly.
    item_factor = (
        escalation_factor.get(severity, DEFAULT_ESCALATION_FACTOR)
        if isinstance(escalation_factor, dict)
        else escalation_factor
    )
    budget_hours = sla_hours.get(severity)
    budget_seconds = budget_hours * 3600.0 if budget_hours is not None else None
    overdue = (
        max(0.0, age_seconds - budget_seconds)
        if age_seconds is not None and budget_seconds is not None
        else 0.0
    )
    # The ladder's second rung: past the item's factor × the budget
    # the breach is no longer fresh — it has been waiting, unacted
    # on, for a whole second budget. The stage names where the item
    # stands so the queue, the summary, and the sweep all read the
    # same ladder.
    escalation_budget_hours = (
        budget_hours * item_factor if budget_hours is not None else None
    )
    escalation_budget_seconds = (
        escalation_budget_hours * 3600.0
        if escalation_budget_hours is not None
        else None
    )
    escalation_overdue = (
        max(0.0, age_seconds - escalation_budget_seconds)
        if age_seconds is not None and escalation_budget_seconds is not None
        else 0.0
    )
    escalated = escalation_overdue > 0
    return {
        "shipment_id": result.shipment_id,
        "exception_type": result.classification.exception_type.value,
        "severity": severity,
        "confidence": result.classification.confidence,
        "carrier": entry.get("carrier", ""),
        "customer_name": entry.get("consignee", ""),
        "origin": entry.get("origin", ""),
        "destination": entry.get("destination", ""),
        "lane": entry.get("lane", ""),
        "created_at": record.created_at,
        "age_seconds": age_seconds,
        "age_bucket": age_bucket(age_seconds),
        # The SLA view for this item: its severity's age budget, and
        # whether (and by how long) the wait has already blown it.
        "sla_hours": budget_hours,
        "sla_breach": overdue > 0,
        "sla_overdue_seconds": round(overdue, 1),
        # The escalation ladder: the second threshold (factor × the
        # budget), the factor that applied to this item's severity,
        # whether the wait has blown that too, and the stage the
        # item stands on — within_budget | breach | escalated.
        "sla_escalation_factor": item_factor,
        "sla_escalation_hours": (
            round(escalation_budget_hours, 3)
            if escalation_budget_hours is not None
            else None
        ),
        "sla_escalated": escalated,
        "sla_escalation_overdue_seconds": round(escalation_overdue, 1),
        "sla_stage": (
            "escalated" if escalated else "breach" if overdue > 0 else "within_budget"
        ),
        "delay_hours": result.delay_hours,
        "flags": flags,
    }


def approval_queue(
    records: list[ApprovalRecord],
    now: datetime | None = None,
    sla_hours: dict | None = None,
    escalation_factor: float | dict | None = None,
) -> list[dict]:
    """Shipments awaiting a decision, severity first, then oldest.

    Only ``awaiting_approval`` records queue — a decided shipment is
    history, not work. Sorting is total and deterministic: severity
    rank, then the analysis timestamp (records without one sort
    first — they are the oldest by definition), then shipment id.
    ``sla_hours`` (severity → age budget in hours) defaults to
    :data:`DEFAULT_SLA_HOURS`; the service passes the env-configured
    thresholds (:func:`sla_thresholds_from_env`).
    ``escalation_factor`` (the ladder's second-rung multiple of the
    budget) is a float applied uniformly or a severity → factor map
    (:func:`sla_escalation_factors_from_env`); it defaults to
    :data:`DEFAULT_ESCALATION_FACTOR`.
    """
    thresholds = sla_hours if sla_hours is not None else DEFAULT_SLA_HOURS
    factor = (
        escalation_factor
        if escalation_factor is not None
        else DEFAULT_ESCALATION_FACTOR
    )
    moment = now or datetime.now(timezone.utc)
    awaiting = [
        record
        for record in records
        if record.result.approval_status == "awaiting_approval"
    ]
    items = [_queue_item(record, moment, thresholds, factor) for record in awaiting]
    items.sort(
        key=lambda item: (
            _SEVERITY_RANK.get(item["severity"], len(_SEVERITY_RANK)),
            item["created_at"] or "",
            item["shipment_id"],
        )
    )
    return items


def queue_summary(items: list[dict]) -> dict:
    """The queue's own health, over :func:`approval_queue` items:
    depth, SLA breaches, and the age/severity mix — the numbers the
    ``/queue`` response and the console headline report."""
    by_bucket = {bucket: 0 for bucket in AGE_BUCKETS}
    by_severity: Counter = Counter()
    breaches = 0
    escalations = 0
    oldest: float | None = None
    for item in items:
        by_bucket[item["age_bucket"]] = by_bucket.get(item["age_bucket"], 0) + 1
        by_severity[item["severity"]] += 1
        if item["sla_breach"]:
            breaches += 1
        if item.get("sla_escalated"):
            escalations += 1
        age = item["age_seconds"]
        if age is not None and (oldest is None or age > oldest):
            oldest = age
    return {
        "total": len(items),
        "sla_breaches": breaches,
        "sla_escalations": escalations,
        "by_bucket": by_bucket,
        "by_severity": dict(sorted(by_severity.items())),
        "oldest_age_seconds": oldest,
    }


# ---------------------------------------------------------------------------
# Carrier scorecards
# ---------------------------------------------------------------------------


def carrier_scorecard(
    records: list[ApprovalRecord],
    carrier: str,
    *,
    exclude_shipment_id: str | None = None,
) -> dict | None:
    """Aggregate one carrier's stored history, or None when they have none.

    Counts come from the same history entries the diagnosis memory
    matches on, so the scorecard and the memory can never disagree
    about the past. Rates are proportions of the carrier's own
    shipments (damage rate) and of the human decisions on them
    (approval rate; ``None`` until a human has decided any).
    """
    hits: list[tuple[ApprovalRecord, dict]] = []
    for record in records:
        entry = history_entry(record)
        if entry is None or not carrier or entry["carrier"] != carrier:
            continue
        if exclude_shipment_id and entry["shipment_id"] == exclude_shipment_id:
            continue
        hits.append((record, entry))
    if not hits:
        return None
    mix: Counter = Counter()
    approvals = rejections = 0
    for record, entry in hits:
        exception = entry["exception_type"]
        if exception != "none":
            mix[exception] += 1
        if record.result.approval_status == "approved":
            approvals += 1
        elif record.result.approval_status == "rejected":
            rejections += 1
    shipments = len(hits)
    exceptions = sum(mix.values())
    decisions = approvals + rejections
    return {
        "carrier": carrier,
        "shipments": shipments,
        "exceptions": exceptions,
        "exception_rate": round(exceptions / shipments, 3),
        "exception_mix": dict(sorted(mix.items())),
        "damage_count": mix.get("damage", 0),
        "damage_rate": round(mix.get("damage", 0) / shipments, 3),
        "decisions": decisions,
        "approvals": approvals,
        "rejections": rejections,
        "approval_rate": round(approvals / decisions, 3) if decisions else None,
    }


def all_carrier_scorecards(records: list[ApprovalRecord]) -> list[dict]:
    """Scorecards for every carrier in the store, busiest first."""
    names = set()
    for record in records:
        entry = history_entry(record)
        if entry is not None and entry["carrier"]:
            names.add(entry["carrier"])
    cards = [carrier_scorecard(records, name) for name in names]
    return sorted(
        (card for card in cards if card is not None),
        key=lambda card: (-card["shipments"], card["carrier"]),
    )


def fleet_baseline(records: list[ApprovalRecord]) -> dict | None:
    """The fleet-wide rates every carrier is compared against.

    Shipment-weighted (total exceptions / total shipments, total
    damage / total shipments) over every stored record with a
    history entry — the baseline a carrier's own rates are read
    against by the scorecard-aware option scorer (``options.py``).
    ``None`` when the store holds no history at all.
    """
    shipments = 0
    exceptions = 0
    damage = 0
    carriers = set()
    for record in records:
        entry = history_entry(record)
        if entry is None:
            continue
        shipments += 1
        carriers.add(entry["carrier"])
        if entry["exception_type"] != "none":
            exceptions += 1
            if entry["exception_type"] == "damage":
                damage += 1
    if not shipments:
        return None
    return {
        "shipments": shipments,
        "carriers": len(carriers),
        "exception_rate": round(exceptions / shipments, 3),
        "damage_rate": round(damage / shipments, 3),
    }


def lane_scorecard(
    records: list[ApprovalRecord],
    carrier: str,
    origin: str,
    destination: str,
    *,
    exclude_shipment_id: str | None = None,
) -> dict | None:
    """One carrier's scorecard on ONE lane, or None when they have
    no history there.

    The carrier scorecard computed over the lane's records only —
    same shape, plus the ``lane`` it describes. This is the dominant
    reliability signal when it is thick enough: a carrier's global
    record can hide a corridor where it keeps breaking freight.
    """
    lane = f"{origin} -> {destination}"
    lane_records = [
        record
        for record in records
        if (history_entry(record) or {}).get("lane") == lane
    ]
    card = carrier_scorecard(
        lane_records, carrier, exclude_shipment_id=exclude_shipment_id
    )
    if card is None:
        return None
    return {**card, "lane": lane}


def lane_baseline(
    records: list[ApprovalRecord],
    origin: str,
    destination: str,
    *,
    exclude_shipment_id: str | None = None,
) -> dict | None:
    """The fleet baseline over ONE lane's shipments, or None.

    The comparison point for the lane scorecard: how ALL carriers
    perform on this corridor. Same shape as :func:`fleet_baseline`,
    plus the ``lane`` it describes.
    """
    lane = f"{origin} -> {destination}"
    lane_records = [
        record
        for record in records
        if (history_entry(record) or {}).get("lane") == lane
        and record.result.shipment_id != exclude_shipment_id
    ]
    baseline = fleet_baseline(lane_records)
    if baseline is None:
        return None
    return {**baseline, "lane": lane}


def format_carrier_scorecard_line(card: dict) -> str:
    """One evidence-style line for a carrier scorecard."""
    line = (
        f"carrier scorecard: {card['shipments']} prior shipment(s) with "
        f"{card['carrier']} in the stored record "
        f"(exception rate {card['exception_rate']}"
    )
    if card["exception_mix"]:
        line += f" — {format_type_counts(card['exception_mix'])}"
    line += f"; damage rate {card['damage_rate']}"
    if card["decisions"]:
        line += (
            f"; {card['approvals']} approved / {card['rejections']} rejected "
            f"of {card['decisions']} decided — approval rate {card['approval_rate']}"
        )
    else:
        line += "; no human decisions yet"
    return line + ")"
