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
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from .store import ApprovalRecord, format_type_counts, history_entry

_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _queue_item(record: ApprovalRecord, now: datetime) -> dict:
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
    return {
        "shipment_id": result.shipment_id,
        "exception_type": result.classification.exception_type.value,
        "severity": result.classification.severity.value,
        "confidence": result.classification.confidence,
        "carrier": entry.get("carrier", ""),
        "customer_name": entry.get("consignee", ""),
        "origin": entry.get("origin", ""),
        "destination": entry.get("destination", ""),
        "lane": entry.get("lane", ""),
        "created_at": record.created_at,
        "age_seconds": (
            round((now - created).total_seconds(), 1) if created else None
        ),
        "delay_hours": result.delay_hours,
        "flags": flags,
    }


def approval_queue(
    records: list[ApprovalRecord], now: datetime | None = None
) -> list[dict]:
    """Shipments awaiting a decision, severity first, then oldest.

    Only ``awaiting_approval`` records queue — a decided shipment is
    history, not work. Sorting is total and deterministic: severity
    rank, then the analysis timestamp (records without one sort
    first — they are the oldest by definition), then shipment id.
    """
    moment = now or datetime.now(timezone.utc)
    awaiting = [
        record
        for record in records
        if record.result.approval_status == "awaiting_approval"
    ]
    items = [_queue_item(record, moment) for record in awaiting]
    items.sort(
        key=lambda item: (
            _SEVERITY_RANK.get(item["severity"], len(_SEVERITY_RANK)),
            item["created_at"] or "",
            item["shipment_id"],
        )
    )
    return items


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
