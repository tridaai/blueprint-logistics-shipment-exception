"""Recovery options: proposed by a template or an LLM, scored by code.

The split is the point. Proposing *what could be done* is judgement —
a deterministic template in the default mode, the LLM in provider
mode. Scoring *what each option is worth* is arithmetic, and a model
never does it: :func:`score_option` computes ETA improvement, added
cost, and SLA impact from the computed facts (delay hours, severity)
with fixed, inspectable formulas, and the recommended option is simply
the highest score. Every number in the console, the packet, and the
draft traces back to this function.

Score (0-100-ish, clamped at 0):
    0.60 * sla_score * 100
  + 0.40 * min(eta_improvement, 48) / 48 * 100
  - 0.25 * min(added_cost, 200) / 200 * 100

On top of that base sits one memory term: the **carrier reliability
adjustment** (:func:`reliability_adjustment`). When the shipment's
carrier has a stored track record (its scorecard, ``insights.py``)
that is worse than the fleet baseline — more damage, more exceptions
per shipment — options that reduce reliance on that carrier's
current handling gain a few points, and the option that simply
waits with the same carrier loses them. The term is small (at most
±``RELIABILITY_MAX_POINTS``), explicit, computed here in code, and
printed on every option it touches (``carrier_reliability_adjustment``)
with its workings in the run's option notes — memory feeding the
decision, never hiding inside it. No track record (or one at/better
than the fleet) means no adjustment: the base formula stands alone,
which is why the golden evals — run without a store — are unchanged.

The term has a second axis: the **lane**. When the carrier has
enough stored history on *this shipment's lane* (its lane
scorecard against the lane's own baseline, ``insights.py``), the
lane figures dominate — a carrier can be reliable everywhere
except one corridor, and the corridor is what this shipment is
about to travel. Below that history threshold the carrier-wide
figures stand, exactly as before.
"""

from __future__ import annotations

from .model_backends import DraftContext
from .schemas import RecoveryOption

OPTION_KINDS = (
    "expedite",
    "reroute",
    "partial_reship",
    "reschedule_appointment",
    "correct_documents",
    "wait_and_monitor",
)

_SEVERITY_WEIGHT = {"low": 1, "medium": 2, "high": 3, "critical": 4}

_TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
    "delay": [
        ("expedite", "Expedite the remaining leg",
         "Upgrade the remaining transport to the fastest available service and re-book the final leg today."),
        ("reroute", "Reroute via the alternate hub",
         "Move the shipment through the alternate regional hub to bypass the current hold."),
        ("wait_and_monitor", "Hold and monitor",
         "Keep the current routing, watch the next carrier scan, and update the customer at the next milestone."),
    ],
    "damage": [
        ("partial_reship", "Ship replacement units for the damaged quantity",
         "Release a replacement shipment for the damaged units while the claim packet is reviewed."),
        ("expedite", "Expedite the replacement shipment",
         "Send the replacement units on the fastest available service once released."),
        ("wait_and_monitor", "Hold pending condition review",
         "Keep the shipment at the terminal until the condition review and photographs are complete."),
    ],
    "document_mismatch": [
        ("correct_documents", "Expedite corrected documents from the shipper",
         "Have the shipper issue corrected documents for the conflicting fields today and re-present them for clearance."),
        ("partial_reship", "Release the undisputed portion now",
         "Deliver the portion of the shipment whose documents agree and hold only the disputed units."),
        ("wait_and_monitor", "Hold pending document correction",
         "Keep the shipment on document hold until the corrected documents arrive."),
    ],
    "missed_appointment": [
        ("reschedule_appointment", "Book the next dock appointment today",
         "Secure the next available delivery appointment with the receiving facility and confirm the window."),
        ("expedite", "Expedite re-delivery",
         "Move the shipment to the front of the re-delivery queue with an upgraded service level."),
        ("wait_and_monitor", "Wait for the standard rebooking",
         "Let the carrier rebook the appointment on its standard cycle and update the customer when confirmed."),
    ],
    "none": [
        ("wait_and_monitor", "Continue as planned",
         "No recovery action is needed; keep the standard tracking updates."),
    ],
}


def score_option(
    kind: str, delay_hours: float | None, severity: str
) -> tuple[float, float, float, float]:
    """Deterministic impact scoring. Returns (eta_h, cost, sla, score)."""
    d = delay_hours or 0.0
    w = _SEVERITY_WEIGHT.get(severity, 2)
    if kind == "expedite":
        eta, cost, sla = min(0.8 * d + 4, 48.0), 60 + 15 * w + 1.2 * d, min(0.55 + 0.10 * w, 0.95)
    elif kind == "reroute":
        eta, cost, sla = min(0.5 * d + 2, 30.0), 25 + 8 * w + 0.6 * d, min(0.45 + 0.10 * w, 0.90)
    elif kind == "partial_reship":
        eta, cost, sla = min(0.4 * d + 6, 36.0), 45 + 12 * w + 0.4 * d, min(0.50 + 0.09 * w, 0.92)
    elif kind == "reschedule_appointment":
        eta, cost, sla = 4 + 2 * w, 10 + 3 * w, min(0.50 + 0.08 * w, 0.90)
    elif kind == "correct_documents":
        eta, cost, sla = 6 + 2 * w, 8 + 2 * w, min(0.60 + 0.07 * w, 0.95)
    else:  # wait_and_monitor
        eta, cost, sla = 0.0, 0.0, max(0.05, 0.55 - 0.008 * d - 0.05 * w)
    score = (
        0.60 * sla * 100
        + 0.40 * (min(eta, 48.0) / 48.0 * 100)
        - 0.25 * (min(cost, 200.0) / 200.0 * 100)
    )
    return round(eta, 2), round(cost, 2), round(sla, 4), round(max(score, 0.0), 2)


# The reliability term's bound, in score points, and the minimum
# stored history before a carrier HAS a track record — a single bad
# shipment is an anecdote, and an anecdote must not move a score.
RELIABILITY_MAX_POINTS = 6.0
RELIABILITY_MIN_SHIPMENTS = 3

# How strongly each option kind answers carrier unreliability: moving
# the freight off the troubled path (reroute) answers it most;
# replacing the goods (partial_reship) and shortening exposure
# (expedite) answer it partly; leaving the shipment with the same
# carrier (wait_and_monitor) is penalised by the same amount. Kinds
# not listed are reliability-neutral (they fix paper, not carriage).
_RELIABILITY_FACTORS = {
    "reroute": 1.0,
    "partial_reship": 0.75,
    "expedite": 0.5,
    "wait_and_monitor": -1.0,
}


def _reliability_basis(
    scorecard: dict | None,
    baseline: dict | None,
    lane_scorecard: dict | None,
    lane_baseline: dict | None,
) -> tuple[dict | None, dict | None, str]:
    """Which figures the reliability term reads, and their scope.

    The lane figures dominate when they exist and the carrier's
    history *on this lane* clears the same minimum-history gate the
    carrier-wide term uses — a thin lane record is an anecdote, and
    the term falls back to the carrier-wide figures rather than
    pricing one. Returns (card, baseline, scope) with scope
    ``"lane"`` or ``"carrier"``.
    """
    if (
        lane_scorecard
        and lane_baseline
        and lane_scorecard.get("shipments", 0) >= RELIABILITY_MIN_SHIPMENTS
    ):
        return lane_scorecard, lane_baseline, "lane"
    return scorecard, baseline, "carrier"


def reliability_adjustment(
    kind: str,
    scorecard: dict | None,
    baseline: dict | None,
    *,
    lane_scorecard: dict | None = None,
    lane_baseline: dict | None = None,
) -> float:
    """The carrier reliability term for one option kind, in points.

    ``unreliability`` is the carrier's excess over the baseline —
    damage excess counts double, because damage is the failure
    waiting cannot undo:

        min(1, 2 * max(0, damage_rate - baseline_damage_rate)
               + max(0, exception_rate - baseline_exception_rate))

    The figures are the lane's when the lane basis is active (see
    :func:`_reliability_basis`), the carrier-wide ones otherwise.
    The term is that index scaled to at most
    ``RELIABILITY_MAX_POINTS`` and signed by the kind's factor.
    Zero when there is no scorecard, no baseline, too little
    history, a reliability-neutral kind, or a carrier performing
    at/better than the comparison — the common case, by design.
    """
    card, base, _scope = _reliability_basis(
        scorecard, baseline, lane_scorecard, lane_baseline
    )
    if not card or not base:
        return 0.0
    if card.get("shipments", 0) < RELIABILITY_MIN_SHIPMENTS:
        return 0.0
    factor = _RELIABILITY_FACTORS.get(kind, 0.0)
    if factor == 0.0:
        return 0.0
    excess_damage = max(0.0, card["damage_rate"] - base["damage_rate"])
    excess_exception = max(0.0, card["exception_rate"] - base["exception_rate"])
    unreliability = min(1.0, 2.0 * excess_damage + excess_exception)
    if unreliability <= 0.0:
        return 0.0
    return round(RELIABILITY_MAX_POINTS * unreliability * factor, 2)


def reliability_note(
    scorecard: dict | None,
    baseline: dict | None,
    *,
    lane_scorecard: dict | None = None,
    lane_baseline: dict | None = None,
) -> str:
    """The one-line workings of a run's reliability term, for the
    trace/console: the numbers the adjustment came from, read at
    whichever basis was active — lane-conditioned when the lane
    figures dominated, carrier-wide otherwise."""
    card, base, scope = _reliability_basis(
        scorecard, baseline, lane_scorecard, lane_baseline
    )
    if scope == "lane" and card is not None and base is not None:
        return (
            f"carrier reliability term applied (lane-conditioned): "
            f"{card['carrier']} on {card.get('lane', '')} runs damage rate "
            f"{card['damage_rate']} / exception rate {card['exception_rate']} "
            f"against the lane's {base['damage_rate']} / "
            f"{base['exception_rate']} over {card['shipments']} prior "
            "shipment(s) on this lane — options that reduce reliance on "
            "this carrier on this lane gain, waiting loses "
            f"(max ±{RELIABILITY_MAX_POINTS:g} points, computed in options.py)"
        )
    card = card if card is not None else scorecard
    base = base if base is not None else baseline
    return (
        f"carrier reliability term applied: {card['carrier']} runs "
        f"damage rate {card['damage_rate']} / exception rate "
        f"{card['exception_rate']} against the fleet's "
        f"{base['damage_rate']} / {base['exception_rate']} "
        f"over {card['shipments']} prior shipment(s) — options "
        "that reduce reliance on this carrier gain, waiting loses "
        f"(max ±{RELIABILITY_MAX_POINTS:g} points, computed in options.py)"
    )


def _template_proposals(exception_type: str) -> list[dict]:
    return [
        {"kind": kind, "title": title, "description": description}
        for kind, title, description in _TEMPLATES.get(exception_type, _TEMPLATES["none"])
    ]


def build_recovery_options(
    *,
    exception_type: str,
    severity: str,
    delay_hours: float | None,
    backend=None,
    context: DraftContext | None = None,
    notes: list[str] | None = None,
    carrier_scorecard: dict | None = None,
    fleet_baseline: dict | None = None,
    carrier_lane_scorecard: dict | None = None,
    lane_baseline: dict | None = None,
) -> list[RecoveryOption]:
    """Propose (template or LLM), validate kinds, score in code, recommend.

    When ``notes`` is given, a provider failure that degraded this node
    to template proposals is appended to it — degradation is recorded,
    never silent. ``carrier_scorecard`` + ``fleet_baseline`` (the
    shipment carrier's stored track record and the fleet's rates, from
    ``insights.py``) feed the reliability term on top of the base
    score — see :func:`reliability_adjustment`; both absent (the eval
    and first-run case) leaves every score exactly the base formula's.
    ``carrier_lane_scorecard`` + ``lane_baseline`` are the same figures
    for this shipment's lane alone: when the lane history is thick
    enough they are the figures the term reads.
    """
    proposals: list[dict] | None = None
    propose_fn = getattr(backend, "propose_options", None) if backend is not None else None
    if propose_fn is not None and context is not None:
        try:
            raw = propose_fn(context)
        except Exception as exc:  # proposals never fail the run
            raw = None
            if notes is not None:
                notes.append(
                    f"LLM option proposals failed ({exc}) — template proposals used"
                )
        if raw:
            seen: set[str] = set()
            valid: list[dict] = []
            for entry in raw:
                kind = entry.get("kind")
                if kind in OPTION_KINDS and kind not in seen:
                    seen.add(kind)
                    valid.append(entry)
            if len(valid) >= 2:
                proposals = valid[:3]
    if proposals is None:
        proposals = _template_proposals(exception_type)

    options: list[RecoveryOption] = []
    reliability_applied = False
    for i, proposal in enumerate(proposals):
        eta, cost, sla, score = score_option(proposal["kind"], delay_hours, severity)
        adjustment = reliability_adjustment(
            proposal["kind"],
            carrier_scorecard,
            fleet_baseline,
            lane_scorecard=carrier_lane_scorecard,
            lane_baseline=lane_baseline,
        )
        if adjustment:
            reliability_applied = True
            score = round(max(score + adjustment, 0.0), 2)
        options.append(
            RecoveryOption(
                option_id=f"OPT-{i + 1}",
                kind=proposal["kind"],
                title=proposal["title"],
                description=proposal["description"],
                eta_improvement_hours=eta,
                added_cost_units=cost,
                sla_score=sla,
                score=score,
                carrier_reliability_adjustment=adjustment,
            )
        )
    if reliability_applied and notes is not None:
        notes.append(
            reliability_note(
                carrier_scorecard,
                fleet_baseline,
                lane_scorecard=carrier_lane_scorecard,
                lane_baseline=lane_baseline,
            )
        )
    if options:
        best = max(range(len(options)), key=lambda i: (options[i].score, -i))
        options[best].recommended = True
    return options
