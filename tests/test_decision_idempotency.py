"""Idempotency keys for human decisions.

Analyze has been safe to retry since round 4; decisions were not —
a double-submitted approval errored on the second call, and an
integrating system that could not tell whether its first call had
landed had no safe way to ask again. An ``Idempotency-Key`` on
approve/reject fixes that: the first decision records the key, a
repeat of the SAME decision under the same key returns the recorded
decision (flagged ``idempotent_replay``) with none of its effects
re-fired — no second webhook dispatch, no duplicated feedback —
the OPPOSITE decision under the spent key is a conflict (409), and
a second decision under a different or absent key keeps the old
422 refusal.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import DecisionConflictError, ShipmentService
from shipment_agent.store import InMemoryStore, SQLiteStore

SHIPMENT = {
    "shipment_id": "DEC-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "carrier": "Synthetic Carrier",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


class _SinkHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        type(self).requests.append({"body": body})
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def sink():
    _SinkHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _SinkHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/hook"
    server.shutdown()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("ACTION_WEBHOOK_URL", raising=False)
    monkeypatch.setattr("shipment_agent.service.load_dotenv", lambda *a, **k: None)


def _service(store=None) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store or InMemoryStore(),
        checkpointer=False,
    )


def _awaiting(service: ShipmentService, shipment_id="DEC-1") -> None:
    service.analyze(dict(SHIPMENT, shipment_id=shipment_id))


# ---------------------------------------------------------------------------
# The replay contract
# ---------------------------------------------------------------------------


def test_repeated_approval_under_one_key_returns_the_recorded_decision():
    service = _service()
    _awaiting(service)
    first = service.approve(
        "DEC-1", approver="ops-lead", reason="go", idempotency_key="dec-1"
    )
    assert first.idempotent_replay is False

    replay = service.approve(
        "DEC-1", approver="ops-lead", reason="go", idempotency_key="dec-1"
    )
    assert replay.idempotent_replay is True
    assert replay.approval_status == "approved"
    assert replay.decided_by == "ops-lead"
    # First write wins: even a replay naming someone else returns
    # the decision as recorded.
    other = service.approve(
        "DEC-1", approver="someone-else", idempotency_key="dec-1"
    )
    assert other.decided_by == "ops-lead"


def test_replay_does_not_redispatch_the_webhook(sink, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    service = _service()
    _awaiting(service)
    service.approve("DEC-1", approver="ops-lead", idempotency_key="dec-1")
    assert len(_SinkHandler.requests) == 1

    service.approve("DEC-1", approver="ops-lead", idempotency_key="dec-1")
    assert len(_SinkHandler.requests) == 1  # still exactly one delivery
    record = service._get_store().get("DEC-1", tenant_id="default")
    assert len(record.dispatch_attempts) == 1


def test_replay_does_not_duplicate_feedback():
    service = _service()
    _awaiting(service)
    service.approve(
        "DEC-1",
        approver="ops-lead",
        reason="customer accepted the revised ETA",
        idempotency_key="dec-1",
    )
    store = service._get_store()
    assert len(store.decision_feedback(tenant_id="default")) == 1

    service.approve("DEC-1", approver="ops-lead", idempotency_key="dec-1")
    assert len(store.decision_feedback(tenant_id="default")) == 1


def test_repeated_rejection_under_one_key_replays_too():
    service = _service()
    _awaiting(service)
    service.reject(
        "DEC-1", reviewer="ops-lead", reason="wrong lane", idempotency_key="dec-9"
    )
    replay = service.reject(
        "DEC-1", reviewer="ops-lead", idempotency_key="dec-9"
    )
    assert replay.idempotent_replay is True
    assert replay.approval_status == "rejected"


# ---------------------------------------------------------------------------
# Conflicts and the old refusals
# ---------------------------------------------------------------------------


def test_opposite_decision_under_a_spent_key_conflicts():
    service = _service()
    _awaiting(service)
    service.approve("DEC-1", approver="ops-lead", idempotency_key="dec-1")
    with pytest.raises(DecisionConflictError):
        service.reject("DEC-1", reviewer="ops-lead", idempotency_key="dec-1")

    _awaiting(service, shipment_id="DEC-2")
    service.reject("DEC-2", reviewer="ops-lead", idempotency_key="dec-2")
    with pytest.raises(DecisionConflictError):
        service.approve("DEC-2", approver="ops-lead", idempotency_key="dec-2")


def test_second_decision_without_the_spent_key_still_refuses():
    service = _service()
    _awaiting(service)
    service.approve("DEC-1", approver="ops-lead", idempotency_key="dec-1")
    # A different key is a different operation: the old 422 refusal.
    with pytest.raises(ValueError, match="not awaiting approval"):
        service.approve("DEC-1", approver="ops-lead", idempotency_key="dec-2")
    # And so is no key at all.
    with pytest.raises(ValueError, match="not awaiting approval"):
        service.approve("DEC-1", approver="ops-lead")


def test_unkeyed_decision_cannot_be_replayed_with_a_key():
    service = _service()
    _awaiting(service)
    service.approve("DEC-1", approver="ops-lead")  # no key recorded
    with pytest.raises(ValueError, match="not awaiting approval"):
        service.approve("DEC-1", approver="ops-lead", idempotency_key="dec-1")


def test_decision_key_survives_a_sqlite_round_trip(tmp_path):
    path = tmp_path / "approvals.db"
    service = _service(store=SQLiteStore(path))
    _awaiting(service)
    service.approve("DEC-1", approver="ops-lead", idempotency_key="dec-1")

    reopened = _service(store=SQLiteStore(path))
    replay = reopened.approve(
        "DEC-1", approver="ops-lead", idempotency_key="dec-1"
    )
    assert replay.idempotent_replay is True


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    service = _service()
    _awaiting(service)
    monkeypatch.setattr(api_module, "service", service)
    return TestClient(api_module.app)


def test_api_replay_returns_200_with_the_replay_header(client):
    first = client.post(
        "/shipments/DEC-1/approve",
        json={"actor": "ops-lead"},
        headers={"Idempotency-Key": "dec-1"},
    )
    assert first.status_code == 200
    assert "x-idempotent-replay" not in first.headers

    replay = client.post(
        "/shipments/DEC-1/approve",
        json={"actor": "ops-lead"},
        headers={"Idempotency-Key": "dec-1"},
    )
    assert replay.status_code == 200
    assert replay.headers["x-idempotent-replay"] == "true"
    assert replay.json()["idempotent_replay"] is True
    assert replay.json()["approval_status"] == "approved"


def test_api_opposite_decision_under_spent_key_is_409(client):
    client.post(
        "/shipments/DEC-1/approve",
        json={"actor": "ops-lead"},
        headers={"Idempotency-Key": "dec-1"},
    )
    conflict = client.post(
        "/shipments/DEC-1/reject",
        json={"actor": "ops-lead"},
        headers={"Idempotency-Key": "dec-1"},
    )
    assert conflict.status_code == 409
    # ...while a different key keeps the long-standing 422.
    refused = client.post(
        "/shipments/DEC-1/reject",
        json={"actor": "ops-lead"},
        headers={"Idempotency-Key": "dec-other"},
    )
    assert refused.status_code == 422
