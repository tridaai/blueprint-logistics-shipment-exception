"""Persistence tests: approvals survive across service instances when
the store is the SQLite file — the in-memory store stays the double.
"""

from __future__ import annotations

import pytest

from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore, SQLiteStore, default_store

DELAY_SHIPMENT = {
    "shipment_id": "PER-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


def _service(store) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(), retriever=KeywordRetriever(), store=store
    )


def test_decision_survives_a_fresh_service_instance(tmp_path):
    db_path = tmp_path / "state.db"
    first = _service(SQLiteStore(db_path))
    result = first.analyze(DELAY_SHIPMENT)
    assert result.approval_status == "awaiting_approval"
    first.approve("PER-1", approver="ops-lead")

    # A brand-new service (a "restart") on the same DB file sees it all.
    second = _service(SQLiteStore(db_path))
    stored = second.get("PER-1")
    assert stored is not None
    assert stored.approval_status == "approved"
    assert stored.classification.exception_type.value == "delay"
    assert stored.external_action_taken is False
    # The decision metadata persisted too, not just the status string.
    record = SQLiteStore(db_path).get("PER-1")
    assert record.approver == "ops-lead"
    assert record.approved is True
    # And the gate still holds after the restart: no re-deciding.
    with pytest.raises(ValueError):
        second.approve("PER-1", approver="someone-else")
    with pytest.raises(ValueError):
        second.reject("PER-1", reviewer="someone-else")


def test_rejection_persists_with_reason(tmp_path):
    db_path = tmp_path / "state.db"
    _service(SQLiteStore(db_path)).analyze(DELAY_SHIPMENT)
    _service(SQLiteStore(db_path)).reject("PER-1", reviewer="qa", reason="Call first")
    record = SQLiteStore(db_path).get("PER-1")
    assert record.result.approval_status == "rejected"
    assert record.rejected_by == "qa"
    assert record.reject_reason == "Call first"


def test_reanalyze_resets_the_decision(tmp_path):
    db_path = tmp_path / "state.db"
    service = _service(SQLiteStore(db_path))
    service.analyze(DELAY_SHIPMENT)
    service.approve("PER-1", approver="ops-lead")
    service.analyze(DELAY_SHIPMENT)  # fresh analysis of the same shipment
    record = SQLiteStore(db_path).get("PER-1")
    assert record.result.approval_status == "awaiting_approval"
    assert record.approved is False
    assert record.approver is None


def test_in_memory_store_is_still_a_working_double():
    service = _service(InMemoryStore())
    service.analyze(DELAY_SHIPMENT)
    approved = service.approve("PER-1", approver="ops-lead")
    assert approved.approval_status == "approved"
    assert service.get("PER-1").approval_status == "approved"


def test_default_store_selection(monkeypatch, tmp_path):
    monkeypatch.setattr("shipment_agent.store.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("STATE_DB_PATH", ":memory:")
    assert isinstance(default_store(), InMemoryStore)
    db_file = tmp_path / "chosen.db"
    monkeypatch.setenv("STATE_DB_PATH", str(db_file))
    store = default_store()
    assert isinstance(store, SQLiteStore)
    assert db_file.exists()  # created eagerly with its schema
