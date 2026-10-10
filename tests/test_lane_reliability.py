"""Lane-conditioned carrier reliability.

Round 4's reliability term read the carrier's global record. But a
carrier can be fine everywhere except one corridor — and the
corridor is the one this shipment is about to travel. When the
carrier has enough stored history on the shipment's lane (>=
``RELIABILITY_MIN_SHIPMENTS``), the lane scorecard vs the lane
baseline is the term's dominant input; below that, the term falls
back to the carrier-wide figures, unchanged. These tests pin the
dominance, the fallback, the note's workings, and the end-to-end
effect through the service.
"""

from __future__ import annotations

import json

from shipment_agent.insights import lane_baseline, lane_scorecard
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.options import (
    RELIABILITY_MAX_POINTS,
    build_recovery_options,
    reliability_adjustment,
    reliability_note,
)
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

BASELINE = {"shipments": 40, "carriers": 4, "exception_rate": 0.3, "damage_rate": 0.1}

# A carrier whose GLOBAL record matches the fleet exactly...
CLEAN_CARD = {
    "carrier": "Corridor Freight",
    "shipments": 10,
    "exceptions": 3,
    "exception_rate": 0.3,
    "exception_mix": {"damage": 1, "delay": 2},
    "damage_count": 1,
    "damage_rate": 0.1,
    "decisions": 0,
    "approvals": 0,
    "rejections": 0,
    "approval_rate": None,
}
# ...but which breaks everything it touches on one lane.
LANE_CARD = {
    **CLEAN_CARD,
    "shipments": 4,
    "exceptions": 4,
    "exception_rate": 1.0,
    "exception_mix": {"damage": 3, "delay": 1},
    "damage_count": 3,
    "damage_rate": 0.75,
    "lane": "Memphis, TN -> Charlotte, NC",
}
LANE_BASELINE = {
    "shipments": 8,
    "carriers": 2,
    "exception_rate": 0.5,
    "damage_rate": 0.375,
    "lane": "Memphis, TN -> Charlotte, NC",
}


def test_lane_figures_dominate_when_the_lane_history_is_thick():
    # Carrier-wide, this carrier is exactly average: no adjustment.
    assert reliability_adjustment("reroute", CLEAN_CARD, BASELINE) == 0.0
    # On the lane it is far worse than the lane's own baseline:
    # excess damage 0.375 (x2) + excess exceptions 0.5 -> capped at 1.
    assert (
        reliability_adjustment(
            "reroute",
            CLEAN_CARD,
            BASELINE,
            lane_scorecard=LANE_CARD,
            lane_baseline=LANE_BASELINE,
        )
        == RELIABILITY_MAX_POINTS
    )
    assert (
        reliability_adjustment(
            "wait_and_monitor",
            CLEAN_CARD,
            BASELINE,
            lane_scorecard=LANE_CARD,
            lane_baseline=LANE_BASELINE,
        )
        == -RELIABILITY_MAX_POINTS
    )


def test_thin_lane_history_falls_back_to_the_carrier_figures():
    thin_lane = {**LANE_CARD, "shipments": 2}
    # The lane anecdote is ignored; the carrier-wide figures (clean)
    # decide, so the adjustment is the carrier-only answer: zero.
    assert (
        reliability_adjustment(
            "reroute",
            CLEAN_CARD,
            BASELINE,
            lane_scorecard=thin_lane,
            lane_baseline=LANE_BASELINE,
        )
        == 0.0
    )
    # And with a dirty carrier card, the fallback is its full value.
    dirty = {**CLEAN_CARD, "damage_rate": 0.6, "exception_rate": 0.9}
    carrier_only = reliability_adjustment("reroute", dirty, BASELINE)
    assert carrier_only > 0
    assert (
        reliability_adjustment(
            "reroute",
            dirty,
            BASELINE,
            lane_scorecard=thin_lane,
            lane_baseline=LANE_BASELINE,
        )
        == carrier_only
    )


def test_lane_note_prints_the_lane_workings():
    note = reliability_note(
        CLEAN_CARD,
        BASELINE,
        lane_scorecard=LANE_CARD,
        lane_baseline=LANE_BASELINE,
    )
    assert "lane-conditioned" in note
    assert "Memphis, TN -> Charlotte, NC" in note
    assert "0.75" in note  # the lane damage rate the term read
    # The carrier-wide note is unchanged when no lane basis is active.
    plain = reliability_note(CLEAN_CARD, BASELINE)
    assert "lane-conditioned" not in plain
    assert plain.startswith("carrier reliability term applied: ")


def test_build_options_uses_the_lane_and_says_so():
    notes: list[str] = []
    options = build_recovery_options(
        exception_type="delay",
        severity="high",
        delay_hours=36.0,
        notes=notes,
        carrier_scorecard=CLEAN_CARD,
        fleet_baseline=BASELINE,
        carrier_lane_scorecard=LANE_CARD,
        lane_baseline=LANE_BASELINE,
    )
    by_kind = {o.kind: o for o in options}
    assert by_kind["reroute"].carrier_reliability_adjustment == RELIABILITY_MAX_POINTS
    assert by_kind["wait_and_monitor"].carrier_reliability_adjustment == (
        -RELIABILITY_MAX_POINTS
    )
    assert any("lane-conditioned" in n for n in notes)


# ---------------------------------------------------------------------------
# End to end: the lane figures come from real stored history
# ---------------------------------------------------------------------------

DAMAGE = "Cartons crushed in transit, goods damaged"
CLEAN = "In transit, on schedule"


def _payload(shipment_id, carrier, origin, destination, event, *, delayed=False):
    return {
        "shipment_id": shipment_id,
        "origin": origin,
        "destination": destination,
        "customer_name": "Synthetic Customer",
        "carrier": carrier,
        "scheduled_delivery": "2026-10-10T09:00:00",
        "estimated_delivery": (
            "2026-10-11T21:00:00" if delayed else "2026-10-10T09:00:00"
        ),
        "latest_event": event,
        "documents": [],
    }


def _service() -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=InMemoryStore(),
        checkpointer=False,
    )


def _seed_corridor_history(service: ShipmentService) -> None:
    """Corridor Freight: wrecks the Memphis->Charlotte lane (3/3
    damage) but is mostly clean elsewhere (6 clean of 9 total), so
    its carrier-wide rates sit near the fleet's while its lane
    rates are the worst on the corridor. Steady Line runs the same
    lane clean, giving the lane a baseline below Corridor's rates."""
    for i in range(3):
        service.analyze(
            _payload(
                f"LANE-D{i}", "Corridor Freight", "Memphis, TN", "Charlotte, NC", DAMAGE
            )
        )
    for i in range(6):
        service.analyze(
            _payload(
                f"LANE-C{i}", "Corridor Freight", "Atlanta, GA", "Charlotte, NC", CLEAN
            )
        )
    for i in range(3):
        service.analyze(
            _payload(f"LANE-S{i}", "Steady Line", "Memphis, TN", "Charlotte, NC", CLEAN)
        )


def test_lane_projections_over_stored_records():
    service = _service()
    _seed_corridor_history(service)
    records = service._get_store().records()

    card = lane_scorecard(records, "Corridor Freight", "Memphis, TN", "Charlotte, NC")
    assert card["shipments"] == 3
    assert card["damage_rate"] == 1.0
    assert card["lane"] == "Memphis, TN -> Charlotte, NC"
    base = lane_baseline(records, "Memphis, TN", "Charlotte, NC")
    assert base["shipments"] == 6
    assert base["damage_rate"] == 0.5
    # A lane the carrier never ran has no card.
    assert (
        lane_scorecard(records, "Corridor Freight", "Memphis, TN", "Nashville, TN")
        is None
    )


def test_new_shipment_on_the_bad_lane_is_priced_on_the_lane():
    service = _service()
    _seed_corridor_history(service)

    result = service.analyze(
        _payload(
            "LANE-NEW",
            "Corridor Freight",
            "Memphis, TN",
            "Charlotte, NC",
            "Delayed at regional hub due to weather hold",
            delayed=True,
        )
    )
    by_kind = {o.kind: o for o in result.recovery_options}
    # The full lane term (+-6), NOT the mild carrier-wide term the
    # same history would have produced before this round (~1.5).
    assert by_kind["reroute"].carrier_reliability_adjustment == RELIABILITY_MAX_POINTS
    assert by_kind["wait_and_monitor"].carrier_reliability_adjustment == (
        -RELIABILITY_MAX_POINTS
    )
    # The workings are printed in the run's trace, lane scope named.
    assert "lane-conditioned" in json.dumps(result.model_dump(mode="json"))


def test_new_shipment_on_an_unrun_lane_keeps_the_carrier_term():
    service = _service()
    _seed_corridor_history(service)

    result = service.analyze(
        _payload(
            "LANE-ELSE",
            "Corridor Freight",
            "Memphis, TN",
            "Nashville, TN",
            "Delayed at regional hub due to weather hold",
            delayed=True,
        )
    )
    by_kind = {o.kind: o for o in result.recovery_options}
    # Memphis->Nashville has no stored history for anyone, so there
    # is no lane basis: the mild carrier-wide term applies (damage
    # excess 0.333 - 0.25 at the scorecard's 3-decimal rounding,
    # exception excess the same -> unreliability 0.249 -> 1.49
    # points), exactly as before this round.
    assert by_kind["reroute"].carrier_reliability_adjustment == 1.49
    assert "lane-conditioned" not in json.dumps(result.model_dump(mode="json"))
