"""Scorecard-aware option scoring — the carrier reliability term.

The base option score is a formula over this shipment's facts. The
reliability term adds the carrier's stored track record: a carrier
running more damage / more exceptions than the fleet baseline makes
the options that reduce reliance on it score higher, and waiting
score lower — by a small, explicit, code-computed amount that is
printed on the option itself. These tests pin the arithmetic, the
gates (no history / thin history / at-baseline carrier → zero), the
one case where the term is decisive, and the eval guarantee: a run
without a store behind it scores exactly the base formula.
"""

from __future__ import annotations

from shipment_agent.insights import fleet_baseline
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.options import (
    RELIABILITY_MAX_POINTS,
    build_recovery_options,
    reliability_adjustment,
    score_option,
)
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import load_sample_shipments
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

BAD_CARD = {
    "carrier": "Rough Freight",
    "shipments": 10,
    "exceptions": 8,
    "exception_rate": 0.8,
    "exception_mix": {"damage": 5, "delay": 3},
    "damage_count": 5,
    "damage_rate": 0.5,
    "decisions": 0,
    "approvals": 0,
    "rejections": 0,
    "approval_rate": None,
}
BASELINE = {"shipments": 40, "carriers": 4, "exception_rate": 0.3, "damage_rate": 0.1}


def _sample(sample_id: str, **overrides) -> dict:
    sample = next(
        s for s in load_sample_shipments() if s["shipment_id"] == sample_id
    )
    return {**sample, **overrides}


# ---------------------------------------------------------------------------
# The term's arithmetic and gates
# ---------------------------------------------------------------------------


def test_reliability_adjustment_math():
    # Excess damage 0.4 (counts double) + excess exceptions 0.5 →
    # unreliability caps at 1.0, so each kind gets factor × the max.
    assert reliability_adjustment("reroute", BAD_CARD, BASELINE) == RELIABILITY_MAX_POINTS
    assert reliability_adjustment("partial_reship", BAD_CARD, BASELINE) == 4.5
    assert reliability_adjustment("expedite", BAD_CARD, BASELINE) == 3.0
    assert reliability_adjustment("wait_and_monitor", BAD_CARD, BASELINE) == -RELIABILITY_MAX_POINTS
    # Reliability-neutral kinds (they fix paper, not carriage).
    assert reliability_adjustment("correct_documents", BAD_CARD, BASELINE) == 0.0
    assert reliability_adjustment("reschedule_appointment", BAD_CARD, BASELINE) == 0.0


def test_reliability_adjustment_scales_with_the_excess():
    card = {**BAD_CARD, "damage_rate": 0.2, "exception_rate": 0.4}
    # Excess damage 0.1 ×2 + excess exception 0.1 = 0.3 → 6 × 0.3 × factor.
    assert reliability_adjustment("reroute", card, BASELINE) == 1.8
    assert reliability_adjustment("expedite", card, BASELINE) == 0.9


def test_reliability_gates():
    # No card, no baseline: no term.
    assert reliability_adjustment("reroute", None, BASELINE) == 0.0
    assert reliability_adjustment("reroute", BAD_CARD, None) == 0.0
    # Too little history: an anecdote is not a track record.
    thin = {**BAD_CARD, "shipments": 2}
    assert reliability_adjustment("reroute", thin, BASELINE) == 0.0
    # A carrier at or better than the fleet: nothing to answer for.
    good = {**BAD_CARD, "damage_rate": 0.05, "exception_rate": 0.2}
    assert reliability_adjustment("reroute", good, BASELINE) == 0.0
    assert reliability_adjustment("wait_and_monitor", good, BASELINE) == 0.0


# ---------------------------------------------------------------------------
# The scorer with and without a track record
# ---------------------------------------------------------------------------


def test_scoring_without_a_card_is_exactly_the_base_formula():
    options = build_recovery_options(
        exception_type="delay", severity="medium", delay_hours=6.0
    )
    for option in options:
        _, _, _, base = score_option(option.kind, 6.0, "medium")
        assert option.score == base
        assert option.carrier_reliability_adjustment == 0.0


def test_the_term_can_be_decisive_and_says_so():
    # A 6-hour medium delay: expedite leads reroute by ~2.6 base
    # points — inside the term's reach. With Rough Freight's record,
    # reroute gains the full 6 and expedite only 3: the
    # recommendation flips, and every option shows its term.
    plain = build_recovery_options(
        exception_type="delay", severity="medium", delay_hours=6.0
    )
    assert next(o for o in plain if o.recommended).kind == "expedite"

    notes: list[str] = []
    adjusted = build_recovery_options(
        exception_type="delay",
        severity="medium",
        delay_hours=6.0,
        carrier_scorecard=BAD_CARD,
        fleet_baseline=BASELINE,
        notes=notes,
    )
    assert next(o for o in adjusted if o.recommended).kind == "reroute"
    for option in adjusted:
        _, _, _, base = score_option(option.kind, 6.0, "medium")
        expected = round(max(base + option.carrier_reliability_adjustment, 0.0), 2)
        assert option.score == expected
    assert any("carrier reliability term applied" in n for n in notes)
    assert "Rough Freight" in notes[-1] if notes else False


def test_scores_never_go_below_zero_under_the_penalty():
    options = build_recovery_options(
        exception_type="delay",
        severity="low",
        delay_hours=1.0,
        carrier_scorecard=BAD_CARD,
        fleet_baseline=BASELINE,
    )
    wait = next(o for o in options if o.kind == "wait_and_monitor")
    assert wait.carrier_reliability_adjustment < 0
    assert wait.score >= 0.0


# ---------------------------------------------------------------------------
# Through the service: the store's history reaches the scorer
# ---------------------------------------------------------------------------


def _service() -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=InMemoryStore(),
        checkpointer=False,
    )


def test_service_run_applies_the_term_from_stored_history():
    service = _service()
    # Rough Freight's past: three damage shipments (SYN-1002's shape)
    # — a damage rate of 1.0 against a fleet that is mostly itself…
    for i in range(3):
        service.analyze(
            _sample("SYN-1002", shipment_id=f"RF-{i}", carrier="Rough Freight")
        )
    # …plus enough clean freight elsewhere to set a low fleet baseline.
    for i in range(6):
        service.analyze(
            _sample("SYN-1011", shipment_id=f"CL-{i}", carrier="Clean Co")
        )
    result = service.analyze(
        _sample("SYN-1001", shipment_id="RF-NEW", carrier="Rough Freight")
    )
    adjustments = [o.carrier_reliability_adjustment for o in result.recovery_options]
    assert any(a != 0.0 for a in adjustments)
    by_kind = {o.kind: o.carrier_reliability_adjustment for o in result.recovery_options}
    if "wait_and_monitor" in by_kind:
        assert by_kind["wait_and_monitor"] < 0


def test_first_time_carrier_scores_the_base_formula():
    service = _service()
    result = service.analyze(_sample("SYN-1001", carrier="Never Seen Lines"))
    assert all(
        o.carrier_reliability_adjustment == 0.0 for o in result.recovery_options
    )


def test_fleet_baseline_math():
    service = _service()
    assert fleet_baseline(service._get_store().records()) is None
    service.analyze(_sample("SYN-1002", shipment_id="B-1", carrier="A Co"))
    service.analyze(_sample("SYN-1011", shipment_id="B-2", carrier="B Co"))
    baseline = fleet_baseline(service._get_store().records())
    assert baseline == {
        "shipments": 2,
        "carriers": 2,
        "exception_rate": 0.5,  # one damage of two shipments
        "damage_rate": 0.5,
    }
