"""Memory tests: prior analysed shipments become diagnosis evidence.

Before diagnosis, the service looks up the store for prior shipments
with the same consignee and the same lane; counts and most recent
exception types join the diagnosis evidence. Works in both stores,
excludes the shipment itself, and stays silent when there is no
history — absence is not evidence.
"""

from __future__ import annotations

from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.schemas import ShipmentInput
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore, SQLiteStore


def _damage(
    shipment_id: str,
    customer: str = "Acme Parts",
    origin: str = "Memphis, TN",
    destination: str = "Charlotte, NC",
) -> dict:
    return {
        "shipment_id": shipment_id,
        "origin": origin,
        "destination": destination,
        "customer_name": customer,
        "latest_event": "Arrived at destination terminal",
        "condition_notes": "Two cartons crushed, contents leaking noted at terminal inspection",
        "documents": [],
    }


def _service(store=None) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store or InMemoryStore(),
    )


def _memory_lines(result) -> list[str]:
    assert result.diagnosis is not None
    return [e for e in result.diagnosis.evidence if e.startswith("memory:")]


def test_first_shipment_has_no_memory_evidence():
    result = _service().analyze(_damage("MEM-1"))
    assert _memory_lines(result) == []


def test_consignee_and_lane_history_accumulate():
    service = _service()
    service.analyze(_damage("MEM-1"))
    second = service.analyze(_damage("MEM-2"))
    lines = _memory_lines(second)
    assert any("1 prior exception(s) for this consignee" in line for line in lines)
    assert any("most recent: damage" in line for line in lines)
    assert any("1 prior exception(s) on this lane" in line for line in lines)
    third = service.analyze(_damage("MEM-3"))
    assert any(
        "2 prior exception(s) for this consignee" in line
        for line in _memory_lines(third)
    )


def test_a_shipment_never_counts_itself():
    service = _service()
    service.analyze(_damage("MEM-1"))
    again = service.analyze(_damage("MEM-1"))  # re-analysis of the same id
    assert _memory_lines(again) == []


def test_lane_only_match_reports_the_lane_line_only():
    service = _service()
    service.analyze(_damage("MEM-1", customer="Acme Parts"))
    other_consignee = service.analyze(_damage("MEM-2", customer="Someone Else"))
    lines = _memory_lines(other_consignee)
    assert any("on this lane" in line for line in lines)
    assert not any("for this consignee" in line for line in lines)


def test_unrelated_shipment_gets_no_memory_lines():
    service = _service()
    service.analyze(_damage("MEM-1"))
    unrelated = service.analyze(
        _damage("MEM-9", customer="Other Co", origin="Atlanta, GA", destination="Miami, FL")
    )
    assert _memory_lines(unrelated) == []


def test_memory_survives_across_service_instances_with_sqlite(tmp_path):
    db = tmp_path / "state.db"
    first = _service(SQLiteStore(db))
    first.analyze(_damage("MEM-1"))
    second = _service(SQLiteStore(db))
    result = second.analyze(_damage("MEM-2"))
    assert any(
        "1 prior exception(s) for this consignee" in line
        for line in _memory_lines(result)
    )


def _carrier_lines(result) -> list[str]:
    assert result.diagnosis is not None
    return [e for e in result.diagnosis.evidence if e.startswith("carrier history:")]


def test_carrier_history_accumulates_across_consignees_and_lanes():
    service = _service()
    service.analyze(_damage("MEM-C1"))
    # Different consignee AND different lane — only the carrier matches.
    second = service.analyze(
        _damage("MEM-C2", customer="Someone Else", origin="Atlanta, GA", destination="Miami, FL")
    )
    lines = _carrier_lines(second)
    assert len(lines) == 1
    assert "1 prior shipment(s) with this carrier" in lines[0]
    assert "Synthetic Carrier" in lines[0]
    assert "exceptions: damage×1" in lines[0]
    # And the consignee/lane memory lines stay absent for this pair.
    memory = _memory_lines(second)
    assert not any("for this consignee" in line for line in memory)
    assert not any("on this lane" in line for line in memory)


def test_carrier_history_counts_by_type():
    service = _service()
    service.analyze(_damage("MEM-C3"))
    delay = {
        "shipment_id": "MEM-C4",
        "origin": "Dallas, TX",
        "destination": "Austin, TX",
        "customer_name": "Other Co",
        "carrier": "Synthetic Carrier",
        "scheduled_delivery": "2026-10-10T09:00:00",
        "estimated_delivery": "2026-10-11T21:00:00",
        "latest_event": "Delayed at regional hub",
        "documents": [],
    }
    service.analyze(delay)
    third = service.analyze(
        _damage("MEM-C5", customer="Third Co", origin="Tampa, FL", destination="Orlando, FL")
    )
    lines = _carrier_lines(third)
    assert len(lines) == 1
    assert "2 prior shipment(s) with this carrier" in lines[0]
    assert "damage×1" in lines[0] and "delay×1" in lines[0]


def test_carrier_history_miss_for_a_new_carrier():
    service = _service()
    service.analyze(_damage("MEM-C6"))
    other = _damage("MEM-C7", customer="Other Co", origin="Atlanta, GA", destination="Miami, FL")
    other["carrier"] = "Another Carrier"
    result = service.analyze(other)
    assert _carrier_lines(result) == []


def test_graph_level_carrier_history_dict_becomes_evidence():
    history = {
        "consignee": "",
        "consignee_count": 0,
        "consignee_recent_types": [],
        "lane": "Memphis, TN -> Charlotte, NC",
        "lane_count": 0,
        "lane_recent_types": [],
        "carrier": "Synthetic Carrier",
        "carrier_count": 3,
        "carrier_exception_count": 3,
        "carrier_type_counts": {"damage": 2, "delay": 1},
    }
    result = run_shipment(ShipmentInput.model_validate(_damage("MEM-C8")), history=history)
    lines = _carrier_lines(result)
    assert len(lines) == 1
    assert "3 prior shipment(s) with this carrier" in lines[0]
    assert "damage×2, delay×1" in lines[0]


def test_graph_level_history_dict_becomes_evidence():
    history = {
        "consignee": "Acme Parts",
        "consignee_count": 2,
        "consignee_recent_types": ["damage", "delay"],
        "lane": "Memphis, TN -> Charlotte, NC",
        "lane_count": 1,
        "lane_recent_types": ["damage"],
    }
    result = run_shipment(ShipmentInput.model_validate(_damage("MEM-7")), history=history)
    lines = _memory_lines(result)
    assert any("2 prior exception(s) for this consignee" in line for line in lines)
    assert any("most recent: damage, delay" in line for line in lines)
    assert any("1 prior exception(s) on this lane" in line for line in lines)
    diagnose_step = next(s for s in result.trace if s.name == "diagnose")
    assert any("memory:" in d for d in diagnose_step.details)
