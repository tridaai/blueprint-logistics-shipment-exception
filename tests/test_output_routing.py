"""Output routing tests — the ACTION_WEBHOOK_URL adapter (off by default).

When an operator configures the webhook, a successful approval POSTs
the approved packet JSON to that URL. A failure is recorded on the
result as dispatch_status=failed and never undoes the approval; a 2xx
records dispatch_status=sent. Unset, nothing is dispatched at all.
Tested against a local stub HTTP server — no external calls.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

SHIPMENT = {
    "shipment_id": "WH-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

RECEIVED: list[dict] = []


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        RECEIVED.append(json.loads(self.rfile.read(length) or b"{}"))
        self.send_response(200 if self.path == "/ok" else 500)
        self.end_headers()

    def log_message(self, *args):  # keep the test output quiet
        pass


@pytest.fixture
def stub_server():
    RECEIVED.clear()
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("ACTION_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ACTION_WEBHOOK_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr("shipment_agent.service.load_dotenv", lambda *a, **k: None)


def _service() -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=InMemoryStore(),
    )


def test_webhook_unset_means_no_dispatch():
    service = _service()
    service.analyze(SHIPMENT)
    result = service.approve("WH-1", approver="ops-lead")
    assert result.approval_status == "approved"
    assert result.dispatch_status is None
    assert result.external_action_taken is False
    assert RECEIVED == []


def test_successful_approval_posts_the_packet(stub_server, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", f"{stub_server}/ok")
    service = _service()
    service.analyze(SHIPMENT)
    result = service.approve("WH-1", approver="ops-lead")
    assert result.dispatch_status == "sent"
    assert result.external_action_taken is True
    assert len(RECEIVED) == 1
    payload = RECEIVED[0]
    assert payload["shipment_id"] == "WH-1"
    assert payload["approval_status"] == "approved"
    assert payload["decided_by"] == "ops-lead"
    assert payload["classification"]["exception_type"] == "delay"
    assert payload["draft"]["subject"]
    assert payload["claim_packet"]["status"] == "draft — not filed"


def test_failed_dispatch_is_recorded_and_approval_stands(stub_server, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", f"{stub_server}/broken")
    service = _service()
    service.analyze(SHIPMENT)
    result = service.approve("WH-1", approver="ops-lead")
    assert result.approval_status == "approved"  # the decision stands
    assert result.dispatch_status == "failed"
    assert result.external_action_taken is False
    stored = service.get("WH-1")
    assert stored is not None
    assert stored.dispatch_status == "failed"


def test_unreachable_webhook_is_failed_not_raised(monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "http://127.0.0.1:1/nowhere")
    monkeypatch.setenv("ACTION_WEBHOOK_TIMEOUT_SECONDS", "2")
    service = _service()
    service.analyze(SHIPMENT)
    result = service.approve("WH-1", approver="ops-lead")
    assert result.approval_status == "approved"
    assert result.dispatch_status == "failed"
