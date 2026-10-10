"""Idempotency keys on analyze.

An integrating system (a TMS, a webhook receiver, a retrying client)
must be able to submit the same analysis twice without paying for it
twice. ``analyze`` with an idempotency key stores the key on the
record; a repeat with the same (key, shipment id) returns the stored
run — flagged ``idempotent_replay``, with the original run's
telemetry — and the pipeline (and its model calls) never happens a
second time. These tests pin that with the mock backend's real call
counter: after the first run, the counter must not move.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore, SQLiteStore

SHIPMENT = {
    "shipment_id": "IDE-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "carrier": "Synthetic Carrier",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


def _service(store=None, backend=None) -> tuple[ShipmentService, MockModelBackend]:
    backend = backend or MockModelBackend()
    service = ShipmentService(
        backend=backend,
        retriever=KeywordRetriever(),
        store=store or InMemoryStore(),
        checkpointer=False,
    )
    return service, backend


def test_repeat_analyze_with_same_key_returns_the_stored_run():
    service, backend = _service()
    first = service.analyze(dict(SHIPMENT), idempotency_key="intake-42")
    calls_after_first = backend.usage_totals()["calls"]
    assert calls_after_first > 0
    assert first.idempotent_replay is False

    second = service.analyze(dict(SHIPMENT), idempotency_key="intake-42")
    # The pipeline did not run again: not one more backend call.
    assert backend.usage_totals()["calls"] == calls_after_first
    assert second.idempotent_replay is True
    # The replay IS the original run, telemetry included — the only
    # difference is the replay flag itself.
    assert second.model_dump(exclude={"idempotent_replay"}) == first.model_dump(
        exclude={"idempotent_replay"}
    )
    assert second.telemetry == first.telemetry
    # And there is still exactly one stored record.
    assert len(service._get_store().records()) == 1


def test_replay_reflects_a_decision_already_taken():
    service, backend = _service()
    service.analyze(dict(SHIPMENT), idempotency_key="intake-7")
    service.approve("IDE-1", approver="ops-lead", reason="go")
    calls = backend.usage_totals()["calls"]

    replay = service.analyze(dict(SHIPMENT), idempotency_key="intake-7")
    assert backend.usage_totals()["calls"] == calls
    assert replay.idempotent_replay is True
    # The stored run is the decided one — a retry never resurrects an
    # awaiting copy or resets the human's decision.
    assert replay.approval_status == "approved"
    assert replay.decided_by == "ops-lead"


def test_a_different_key_or_shipment_reruns():
    service, backend = _service()
    service.analyze(dict(SHIPMENT), idempotency_key="key-a")
    calls = backend.usage_totals()["calls"]

    other_key = service.analyze(dict(SHIPMENT), idempotency_key="key-b")
    assert other_key.idempotent_replay is False
    assert backend.usage_totals()["calls"] > calls
    calls = backend.usage_totals()["calls"]

    other_shipment = service.analyze(
        dict(SHIPMENT, shipment_id="IDE-2"), idempotency_key="key-a"
    )
    assert other_shipment.idempotent_replay is False
    assert backend.usage_totals()["calls"] > calls


def test_no_key_always_reruns_and_blank_key_is_no_key():
    service, backend = _service()
    service.analyze(dict(SHIPMENT))
    calls = backend.usage_totals()["calls"]
    again = service.analyze(dict(SHIPMENT))
    assert again.idempotent_replay is False
    assert backend.usage_totals()["calls"] > calls

    blank = service.analyze(dict(SHIPMENT), idempotency_key="   ")
    assert blank.idempotent_replay is False


def test_key_survives_a_store_round_trip(tmp_path):
    """The dedupe lives in the store, not the process: a fresh
    service over the same SQLite file replays without running."""
    store = SQLiteStore(tmp_path / "state.db")
    service, backend = _service(store=store)
    service.analyze(dict(SHIPMENT), idempotency_key="durable-1")
    assert backend.usage_totals()["calls"] > 0

    fresh_backend = MockModelBackend()
    fresh, _ = _service(store=SQLiteStore(tmp_path / "state.db"), backend=fresh_backend)
    replay = fresh.analyze(dict(SHIPMENT), idempotency_key="durable-1")
    assert replay.idempotent_replay is True
    assert fresh_backend.usage_totals()["calls"] == 0


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    service, _ = _service()
    monkeypatch.setattr(api_module, "service", service)
    return TestClient(api_module.app), service


def test_analyze_endpoint_honours_the_header(client):
    test_client, _ = client
    first = test_client.post(
        "/shipments/analyze",
        json=SHIPMENT,
        headers={"Idempotency-Key": "req-1"},
    )
    assert first.status_code == 200
    assert first.json()["idempotent_replay"] is False
    assert "x-idempotent-replay" not in first.headers

    second = test_client.post(
        "/shipments/analyze",
        json=SHIPMENT,
        headers={"Idempotency-Key": "req-1"},
    )
    assert second.status_code == 200
    assert second.json()["idempotent_replay"] is True
    assert second.headers.get("x-idempotent-replay") == "true"
    assert second.json()["telemetry"] == first.json()["telemetry"]


def test_stream_endpoint_replays_as_a_single_completed_event(client):
    test_client, _ = client
    test_client.post(
        "/shipments/analyze", json=SHIPMENT, headers={"Idempotency-Key": "req-9"}
    )
    with test_client.stream(
        "POST",
        "/shipments/analyze/stream",
        json=SHIPMENT,
        headers={"Idempotency-Key": "req-9"},
    ) as response:
        assert response.status_code == 200
        events = [line for line in response.iter_lines() if line.startswith("data:")]
    assert len(events) == 1
    import json

    payload = json.loads(events[0][len("data:"):])
    assert payload["type"] == "run_completed"
    assert payload["result"]["idempotent_replay"] is True
    assert payload["result"]["shipment_id"] == "IDE-1"
