"""Webhook delivery ledger + retries.

Every approval-webhook dispatch attempt is recorded on the record —
timestamp, outcome, HTTP status, signature id, error, and when the
next retry falls due — and a failed delivery can be retried under a
bounded attempt budget with exponential backoff. The sink is a fake:
a local HTTP stub whose failure count the test controls, so "endpoint
down, endpoint fixed, retry delivers" is exercised end to end with
no external calls.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import (
    ShipmentService,
    sign_webhook_body,
    webhook_retry_delay,
)
from shipment_agent.store import InMemoryStore, SQLiteStore

SHIPMENT = {
    "shipment_id": "LED-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


class _SinkHandler(BaseHTTPRequestHandler):
    """A fake webhook sink: fails the next N POSTs with HTTP 500,
    then accepts. Every request (body + signature header) is kept."""

    failures_remaining = 0
    requests: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        type(self).requests.append(
            {"body": body, "signature": self.headers.get("X-Trida-Signature")}
        )
        if type(self).failures_remaining > 0:
            type(self).failures_remaining -= 1
            self.send_response(500)
        else:
            self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # keep the test output quiet
        pass


@pytest.fixture
def sink():
    _SinkHandler.failures_remaining = 0
    _SinkHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _SinkHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/hook"
    server.shutdown()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "ACTION_WEBHOOK_URL",
        "ACTION_WEBHOOK_SECRET",
        "ACTION_WEBHOOK_TIMEOUT_SECONDS",
        "ACTION_WEBHOOK_MAX_ATTEMPTS",
        "ACTION_WEBHOOK_RETRY_BASE_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.service.load_dotenv", lambda *a, **k: None)


def _service(store=None) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store or InMemoryStore(),
        checkpointer=False,
    )


def _approved(service: ShipmentService, shipment_id: str = "LED-1"):
    service.analyze(dict(SHIPMENT, shipment_id=shipment_id))
    return service.approve(shipment_id, approver="ops-lead")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


# ---------------------------------------------------------------------------
# The ledger itself
# ---------------------------------------------------------------------------


def test_successful_dispatch_is_ledgered(sink, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    service = _service()
    result = _approved(service)
    assert result.dispatch_status == "sent"
    ledger = service.dispatch_ledger("LED-1")
    assert ledger["attempts_used"] == 1
    assert ledger["attempts_remaining"] == 2  # default budget is 3
    assert ledger["next_retry_at"] is None
    (attempt,) = ledger["attempts"]
    assert attempt["attempt"] == 1
    assert attempt["outcome"] == "sent"
    assert attempt["http_status"] == 200
    assert attempt["error"] is None
    assert attempt["signature_id"] is None  # unsigned: no secret configured
    assert _parse(attempt["at"]) <= datetime.now(timezone.utc)


def test_failed_dispatch_ledgers_status_error_and_backoff(sink, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    monkeypatch.setenv("ACTION_WEBHOOK_RETRY_BASE_SECONDS", "10")
    _SinkHandler.failures_remaining = 5
    service = _service()
    result = _approved(service)
    assert result.dispatch_status == "failed"
    assert result.external_action_taken is False
    ledger = service.dispatch_ledger("LED-1")
    (attempt,) = ledger["attempts"]
    assert attempt["outcome"] == "failed"
    assert attempt["http_status"] == 500
    assert attempt["error"] == "HTTP 500"
    # Backoff bookkeeping: the next retry is due one base delay later.
    delta = _parse(attempt["next_retry_at"]) - _parse(attempt["at"])
    assert abs(delta.total_seconds() - 10.0) < 0.01
    assert ledger["next_retry_at"] == attempt["next_retry_at"]


def test_signature_id_is_ledgered_when_signing(sink, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", "s3cret")
    service = _service()
    _approved(service)
    (attempt,) = service.dispatch_ledger("LED-1")["attempts"]
    sent = _SinkHandler.requests[0]
    assert attempt["signature_id"] == sent["signature"]
    assert attempt["signature_id"] == sign_webhook_body(sent["body"], "s3cret")
    assert json.loads(sent["body"])["shipment_id"] == "LED-1"


def test_unreachable_endpoint_ledgers_a_transport_error(monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "http://127.0.0.1:1/nowhere")
    monkeypatch.setenv("ACTION_WEBHOOK_TIMEOUT_SECONDS", "2")
    service = _service()
    result = _approved(service)
    assert result.dispatch_status == "failed"
    (attempt,) = service.dispatch_ledger("LED-1")["attempts"]
    assert attempt["http_status"] is None
    assert "URLError" in attempt["error"]


def test_no_webhook_configured_means_an_empty_ledger(monkeypatch):
    service = _service()
    result = _approved(service)
    assert result.dispatch_status is None
    ledger = service.dispatch_ledger("LED-1")
    assert ledger["attempts"] == []
    assert ledger["dispatch_status"] is None
    assert service.dispatch_ledger("NOPE-404") is None


def test_ledger_survives_a_store_round_trip(sink, monkeypatch, tmp_path):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    _SinkHandler.failures_remaining = 1
    store = SQLiteStore(tmp_path / "state.db")
    service = _service(store=store)
    _approved(service)
    reloaded = SQLiteStore(tmp_path / "state.db").get("LED-1")
    assert reloaded is not None
    assert reloaded.dispatch_status == "failed"
    assert len(reloaded.dispatch_attempts) == 1
    assert reloaded.dispatch_attempts[0]["http_status"] == 500


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


def test_retry_delivers_once_the_sink_recovers(sink, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    monkeypatch.setenv("ACTION_WEBHOOK_RETRY_BASE_SECONDS", "10")
    _SinkHandler.failures_remaining = 1  # approval attempt fails, retry lands
    service = _service()
    _approved(service)
    result = service.retry_dispatch("LED-1", force=True)
    assert result.dispatch_status == "sent"
    assert result.external_action_taken is True
    ledger = service.dispatch_ledger("LED-1")
    assert [a["outcome"] for a in ledger["attempts"]] == ["failed", "sent"]
    assert ledger["attempts"][1]["http_status"] == 200
    assert ledger["attempts"][1]["next_retry_at"] is None
    assert len(_SinkHandler.requests) == 2


def test_retry_respects_the_backoff_unless_forced(sink, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    monkeypatch.setenv("ACTION_WEBHOOK_RETRY_BASE_SECONDS", "3600")
    _SinkHandler.failures_remaining = 5
    service = _service()
    _approved(service)
    with pytest.raises(ValueError, match="Backoff has not elapsed"):
        service.retry_dispatch("LED-1")
    assert len(_SinkHandler.requests) == 1  # no attempt was made
    service.retry_dispatch("LED-1", force=True)
    assert len(_SinkHandler.requests) == 2


def test_second_failure_doubles_the_backoff(sink, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    monkeypatch.setenv("ACTION_WEBHOOK_RETRY_BASE_SECONDS", "10")
    _SinkHandler.failures_remaining = 5
    service = _service()
    _approved(service)
    service.retry_dispatch("LED-1", force=True)
    attempts = service.dispatch_ledger("LED-1")["attempts"]
    assert len(attempts) == 2
    delta = _parse(attempts[1]["next_retry_at"]) - _parse(attempts[1]["at"])
    assert abs(delta.total_seconds() - 20.0) < 0.01


def test_retry_budget_is_bounded(sink, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    monkeypatch.setenv("ACTION_WEBHOOK_MAX_ATTEMPTS", "2")
    _SinkHandler.failures_remaining = 9
    service = _service()
    _approved(service)
    service.retry_dispatch("LED-1", force=True)
    ledger = service.dispatch_ledger("LED-1")
    assert ledger["attempts_used"] == 2
    assert ledger["attempts_remaining"] == 0
    assert ledger["attempts"][-1]["next_retry_at"] is None  # nothing further due
    with pytest.raises(ValueError, match="budget exhausted"):
        service.retry_dispatch("LED-1", force=True)
    assert service.due_dispatch_retries() == []


def test_retry_refuses_a_sent_delivery(sink, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    service = _service()
    _approved(service)
    with pytest.raises(ValueError, match="already delivered"):
        service.retry_dispatch("LED-1", force=True)


def test_retry_refuses_without_a_webhook_or_approval():
    service = _service()
    service.analyze(SHIPMENT)
    with pytest.raises(ValueError, match="not approved"):
        service.retry_dispatch("LED-1", force=True)
    service.approve("LED-1", approver="ops-lead")  # no URL configured
    with pytest.raises(ValueError, match="not configured"):
        service.retry_dispatch("LED-1", force=True)
    with pytest.raises(KeyError):
        service.retry_dispatch("NOPE-404", force=True)


def test_due_dispatch_retries_follows_the_backoff(sink, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    _SinkHandler.failures_remaining = 5
    service = _service()
    # Base 0: the retry is due immediately.
    monkeypatch.setenv("ACTION_WEBHOOK_RETRY_BASE_SECONDS", "0")
    _approved(service, "LED-1")
    assert service.due_dispatch_retries() == ["LED-1"]
    # A long base pushes the due time out; an explicit clock past it
    # makes it due again.
    monkeypatch.setenv("ACTION_WEBHOOK_RETRY_BASE_SECONDS", "3600")
    _SinkHandler.failures_remaining = 5
    _approved(service, "LED-2")
    assert service.due_dispatch_retries() == ["LED-1"]
    later = datetime.now(timezone.utc) + timedelta(hours=2)
    assert set(service.due_dispatch_retries(now=later)) == {"LED-1", "LED-2"}


def test_backoff_schedule_is_exponential(monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_RETRY_BASE_SECONDS", "15")
    assert webhook_retry_delay(1) == 15.0
    assert webhook_retry_delay(2) == 30.0
    assert webhook_retry_delay(3) == 60.0


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch, sink):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    monkeypatch.setenv("ACTION_WEBHOOK_RETRY_BASE_SECONDS", "3600")
    _SinkHandler.failures_remaining = 1  # first approval dispatch fails
    service = _service()
    service.analyze(SHIPMENT)
    service.approve("LED-1", approver="ops-lead")
    monkeypatch.setattr(api_module, "service", service)
    return TestClient(api_module.app)


def test_dispatch_ledger_endpoint(client):
    response = client.get("/shipments/LED-1/dispatch")
    assert response.status_code == 200
    body = response.json()
    assert body["dispatch_status"] == "failed"
    assert body["attempts_used"] == 1
    assert body["attempts"][0]["http_status"] == 500
    assert body["next_retry_at"]


def test_dispatch_ledger_endpoint_404s_for_unknown_shipment(client):
    assert client.get("/shipments/NOPE-404/dispatch").status_code == 404


def test_dispatch_retry_endpoint(client):
    blocked = client.post("/shipments/LED-1/dispatch/retry", json={})
    assert blocked.status_code == 422  # backoff (1h base) has not elapsed
    forced = client.post("/shipments/LED-1/dispatch/retry", json={"force": True})
    assert forced.status_code == 200
    assert forced.json()["dispatch_status"] == "sent"
    ledger = client.get("/shipments/LED-1/dispatch").json()
    assert [a["outcome"] for a in ledger["attempts"]] == ["failed", "sent"]
    again = client.post("/shipments/LED-1/dispatch/retry", json={"force": True})
    assert again.status_code == 422  # already delivered
