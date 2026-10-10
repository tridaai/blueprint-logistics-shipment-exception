"""Approval queue + carrier scorecards — projections of the store.

The queue is the approver's worklist: awaiting shipments sorted by
severity then age, with the flags that change how a case is read
(cross-check disagreement, repair, reviewer block, guardrails,
information needed). The scorecards aggregate each carrier's stored
history — exception mix, damage rate, human approval rate — and the
scorecard for a new shipment's carrier joins its diagnosis evidence.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.insights import approval_queue
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import load_sample_shipments
from shipment_agent.schemas import ClassificationCrossCheck
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore


def _sample(sample_id: str, **overrides) -> dict:
    sample = next(
        s for s in load_sample_shipments() if s["shipment_id"] == sample_id
    )
    return {**sample, **overrides}


def _service(store=None) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store or InMemoryStore(),
        checkpointer=False,
    )


def _set_created_at(service: ShipmentService, shipment_id: str, when: str) -> None:
    store = service._get_store()
    record = store.get(shipment_id)
    assert record is not None
    record.created_at = when
    store.save(record)


# ---------------------------------------------------------------------------
# Approval queue
# ---------------------------------------------------------------------------


def test_queue_sorts_by_severity_then_age_and_skips_decided():
    service = _service()
    base = datetime(2026, 10, 10, 8, 0, tzinfo=timezone.utc)
    # Analysed in scrambled order; created_at is then pinned so the
    # expected order is fully determined by (severity, age).
    for sid in ("SYN-1005", "SYN-1002", "SYN-1006", "SYN-1003", "SYN-1001", "SYN-1004"):
        service.analyze(_sample(sid))
    ages = {
        "SYN-1006": 0,  # critical — first regardless of age
        "SYN-1001": 1,  # high, earliest of the highs
        "SYN-1002": 3,  # high
        "SYN-1003": 5,  # high, latest
        "SYN-1004": 9,  # medium — after every high
        "SYN-1005": 12,  # low — last
    }
    for sid, hours in ages.items():
        _set_created_at(service, sid, (base + timedelta(hours=hours)).isoformat())
    service.approve("SYN-1004", approver="ops-lead")  # decided: leaves the queue

    queue = service.approval_queue()
    assert [item["shipment_id"] for item in queue] == [
        "SYN-1006",
        "SYN-1001",
        "SYN-1002",
        "SYN-1003",
        "SYN-1005",
    ]
    top = queue[0]
    assert top["severity"] == "critical"
    assert top["exception_type"] == "delay"
    assert top["lane"] == "Birmingham, AL -> Knoxville, TN"
    assert top["carrier"] == "Synthetic Carrier"
    later = base + timedelta(hours=13)
    items = approval_queue(service._get_store().records(), now=later)
    assert items[0]["age_seconds"] == 13 * 3600  # SYN-1006 pinned at base


def test_queue_flags_surface_the_cases_that_need_a_closer_look():
    service = _service()
    # SYN-1013: the carrier note promises a refund — the template draft
    # inherits it, guardrails fail, repair is attempted.
    service.analyze(_sample("SYN-1013"))
    (item,) = service.approval_queue()
    assert item["flags"]["guardrails_passed"] is False
    assert item["flags"]["repair_attempted"] is True
    assert item["flags"]["repaired"] is False
    assert item["flags"]["cross_check_disagreement"] is False  # mock: rules_only

    # A cross-check disagreement (provider mode) flags the item too.
    store = service._get_store()
    record = store.get("SYN-1013")
    record.result = record.result.model_copy(
        update={
            "cross_check": ClassificationCrossCheck(
                rule_exception_type="delay",
                rule_severity="high",
                rule_confidence=0.94,
                llm_exception_type="damage",
                llm_severity="high",
                llm_confidence=0.7,
                agrees=False,
                resolution="rules_authoritative",
                adopted_source="rules",
            )
        }
    )
    store.save(record)
    (item,) = service.approval_queue()
    assert item["flags"]["cross_check_disagreement"] is True
    assert item["flags"]["cross_check_resolution"] == "rules_authoritative"


def test_queue_endpoint_shape():
    service = _service()
    service.analyze(_sample("SYN-1001"))
    service.analyze(_sample("SYN-1005"))
    api_service = service
    import shipment_agent.api as api

    original = api.service
    api.service = api_service
    try:
        client = TestClient(api.app)
        response = client.get("/queue")
    finally:
        api.service = original
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["queue"][0]["shipment_id"] == "SYN-1001"  # high before low
    assert body["queue"][0]["flags"]["guardrails_passed"] is True


# ---------------------------------------------------------------------------
# Carrier scorecards
# ---------------------------------------------------------------------------


def _acme_service() -> ShipmentService:
    """Three Acme Freight shipments: a damage (approved), a delay
    (awaiting), a clean delivery (rejected); two clean Clean Co runs."""
    service = _service()
    service.analyze(_sample("SYN-1002", shipment_id="AC-1", carrier="Acme Freight"))
    service.approve("AC-1", approver="ops-lead")
    service.analyze(_sample("SYN-1001", shipment_id="AC-2", carrier="Acme Freight"))
    service.analyze(_sample("SYN-1011", shipment_id="AC-3", carrier="Acme Freight"))
    service.reject("AC-3", reviewer="ops-lead", reason="duplicate paperwork")
    service.analyze(_sample("SYN-1011", shipment_id="CC-1", carrier="Clean Co"))
    service.analyze(_sample("SYN-1012", shipment_id="CC-2", carrier="Clean Co"))
    return service


def test_carrier_scorecard_math():
    service = _acme_service()
    card = service.carrier_scorecard("Acme Freight")
    assert card is not None
    assert card["shipments"] == 3
    assert card["exceptions"] == 2
    assert card["exception_mix"] == {"damage": 1, "delay": 1}
    assert card["exception_rate"] == 0.667
    assert card["damage_count"] == 1
    assert card["damage_rate"] == 0.333
    assert card["decisions"] == 2
    assert card["approvals"] == 1
    assert card["rejections"] == 1
    assert card["approval_rate"] == 0.5

    clean = service.carrier_scorecard("Clean Co")
    assert clean["shipments"] == 2
    assert clean["exceptions"] == 0
    assert clean["exception_rate"] == 0.0
    assert clean["damage_rate"] == 0.0
    assert clean["decisions"] == 0
    assert clean["approval_rate"] is None

    assert service.carrier_scorecard("Never Seen Co") is None
    cards = service.carrier_scorecards()
    assert [c["carrier"] for c in cards] == ["Acme Freight", "Clean Co"]


def test_scorecard_joins_the_diagnosis_evidence():
    service = _acme_service()
    result = service.analyze(
        _sample("SYN-1005", shipment_id="AC-9", carrier="Acme Freight")
    )
    lines = [
        e for e in result.diagnosis.evidence if e.startswith("carrier scorecard:")
    ]
    assert len(lines) == 1
    assert "3 prior shipment(s) with Acme Freight" in lines[0]
    assert "damage rate 0.333" in lines[0]
    assert "approval rate 0.5" in lines[0]


def test_scorecard_line_appears_even_for_a_clean_carrier():
    service = _acme_service()
    result = service.analyze(
        _sample("SYN-1005", shipment_id="CC-9", carrier="Clean Co")
    )
    lines = [
        e for e in result.diagnosis.evidence if e.startswith("carrier scorecard:")
    ]
    assert len(lines) == 1
    assert "exception rate 0.0" in lines[0]
    assert "no human decisions yet" in lines[0]


def test_no_scorecard_line_for_a_first_time_carrier():
    service = _service()
    result = service.analyze(_sample("SYN-1001", carrier="First Timer Freight"))
    assert not any(
        e.startswith("carrier scorecard:") for e in result.diagnosis.evidence
    )


@pytest.fixture()
def scorecard_client(monkeypatch):
    monkeypatch.setattr(api_module, "service", _acme_service())
    return TestClient(api_module.app)


def test_scorecard_endpoints(scorecard_client):
    response = scorecard_client.get("/carriers/scorecards")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["carriers"][0]["carrier"] == "Acme Freight"

    one = scorecard_client.get("/carriers/Acme Freight/scorecard")
    assert one.status_code == 200
    assert one.json()["damage_rate"] == 0.333

    missing = scorecard_client.get("/carriers/Never Seen Co/scorecard")
    assert missing.status_code == 404
