"""Signed output routing: ACTION_WEBHOOK_SECRET adds an HMAC signature.

When the secret is configured, each webhook delivery carries
``X-Trida-Signature: sha256=<HMAC-SHA256 hex of the raw body>`` so the
receiving system can verify the packet came from the agent and was
not altered. Unsigned when the secret is unset — the pre-existing
behaviour, pinned by the first test here and by test_output_routing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService, sign_webhook_body
from shipment_agent.store import InMemoryStore

SHIPMENT = {
    "shipment_id": "WH-SIG-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

RECEIVED: list[dict] = []


class _CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        RECEIVED.append(
            {"body": body, "signature": self.headers.get("X-Trida-Signature")}
        )
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # keep the test output quiet
        pass


@pytest.fixture
def stub_server():
    RECEIVED.clear()
    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("ACTION_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ACTION_WEBHOOK_SECRET", raising=False)
    monkeypatch.setattr("shipment_agent.service.load_dotenv", lambda *a, **k: None)


def _approved(service: ShipmentService):
    service.analyze(SHIPMENT)
    return service.approve("WH-SIG-1", approver="ops-lead")


def _service() -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=InMemoryStore(),
        checkpointer=False,
    )


def test_sign_webhook_body_format():
    expected = "sha256=" + hmac.new(b"s3cret", b"{}", hashlib.sha256).hexdigest()
    assert sign_webhook_body(b"{}", "s3cret") == expected


def test_delivery_is_signed_when_secret_configured(stub_server, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", stub_server + "/hook")
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", "s3cret")
    result = _approved(_service())
    assert result.dispatch_status == "sent"
    assert len(RECEIVED) == 1
    delivery = RECEIVED[0]
    # The signature must verify over the exact bytes that were sent.
    assert delivery["signature"] == sign_webhook_body(delivery["body"], "s3cret")
    assert json.loads(delivery["body"])["shipment_id"] == "WH-SIG-1"


def test_delivery_is_unsigned_without_secret(stub_server, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", stub_server + "/hook")
    result = _approved(_service())
    assert result.dispatch_status == "sent"
    assert RECEIVED[0]["signature"] is None


def test_wrong_secret_fails_verification(stub_server, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", stub_server + "/hook")
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", "s3cret")
    _approved(_service())
    delivery = RECEIVED[0]
    assert delivery["signature"] != sign_webhook_body(delivery["body"], "other")
