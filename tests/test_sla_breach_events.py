"""SLA breach events: the queue's alarm, on the signed webhook.

The approval queue has flagged SLA breaches since round 4; this
round makes a breach *act*: the first time the sweep observes one,
it fires a single signed ``sla_breach`` webhook event — ledgered on
the record's own SLA ledger (never the approval packet's), deduped
by the record's ``sla_breach_event_at`` marker so it pages once per
shipment per analysis, and opt-in like every external action here
(``SLA_BREACH_WEBHOOK=on``). The sink is the same fake HTTP stub the
webhook-ledger tests use.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.events import CollectingSink
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService, sign_webhook_body
from shipment_agent.store import InMemoryStore, SQLiteStore

SHIPMENT = {
    "shipment_id": "SLA-1",
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

    def log_message(self, *args):
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
        "SLA_BREACH_WEBHOOK",
        "SLA_BREACH_WEBHOOK_URL",
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


def _breaching(service: ShipmentService, shipment_id="SLA-1", tenant_id=None):
    """Analyse a shipment and age its record past every SLA budget."""
    kwargs = {"tenant_id": tenant_id} if tenant_id else {}
    service.analyze(dict(SHIPMENT, shipment_id=shipment_id), **kwargs)
    store = service._get_store()
    record = store.get(shipment_id, tenant_id=tenant_id or "default")
    record.created_at = (
        datetime.now(timezone.utc) - timedelta(days=30)
    ).isoformat()
    store.save(record)
    return record


def _enable(monkeypatch, sink, secret=None):
    monkeypatch.setenv("SLA_BREACH_WEBHOOK", "on")
    monkeypatch.setenv("SLA_BREACH_WEBHOOK_URL", sink)
    if secret:
        monkeypatch.setenv("ACTION_WEBHOOK_SECRET", secret)


# ---------------------------------------------------------------------------
# Firing + dedupe
# ---------------------------------------------------------------------------


def test_sweep_fires_one_signed_event_for_a_new_breach(sink, monkeypatch):
    _enable(monkeypatch, sink, secret="s3cret")
    service = _service()
    _breaching(service)

    entries = service.sla_breach_sweep()

    assert len(entries) == 1
    assert entries[0]["outcome"] == "sent"
    assert entries[0]["shipment_id"] == "SLA-1"
    assert len(_SinkHandler.requests) == 1
    request = _SinkHandler.requests[0]
    body = json.loads(request["body"])
    assert body["event"] == "sla_breach"
    assert body["shipment_id"] == "SLA-1"
    assert body["tenant_id"] == "default"
    assert body["overdue_seconds"] > 0
    # Signed with the shared secret, exactly like the approval packet.
    assert request["signature"] == sign_webhook_body(request["body"], "s3cret")
    assert entries[0]["signature_id"] == request["signature"]


def test_second_sweep_does_not_refire(sink, monkeypatch):
    _enable(monkeypatch, sink)
    service = _service()
    _breaching(service)

    assert len(service.sla_breach_sweep()) == 1
    assert service.sla_breach_sweep() == []
    assert len(_SinkHandler.requests) == 1

    record = service._get_store().get("SLA-1", tenant_id="default")
    assert record.sla_breach_event_at is not None
    assert len(record.sla_dispatch_attempts) == 1
    assert record.sla_dispatch_attempts[0]["outcome"] == "sent"


def test_sla_ledger_never_touches_the_approval_ledger(sink, monkeypatch):
    _enable(monkeypatch, sink)
    service = _service()
    record = _breaching(service)

    service.sla_breach_sweep()

    assert record.sla_dispatch_attempts  # the event's own ledger filled
    assert record.dispatch_attempts == []  # the approval's did not
    assert record.dispatch_status is None
    assert record.result.external_action_taken is False


def test_failed_delivery_is_ledgered_and_still_fires_only_once(sink, monkeypatch):
    _SinkHandler.failures_remaining = 5
    _enable(monkeypatch, sink)
    service = _service()
    record = _breaching(service)

    entries = service.sla_breach_sweep()
    assert entries[0]["outcome"] == "failed"
    assert entries[0]["error"] == "HTTP 500"
    assert record.sla_dispatch_attempts[0]["http_status"] == 500
    # The event fired once; the sweep is an alarm, not a retry loop.
    assert record.sla_breach_event_at is not None
    assert service.sla_breach_sweep() == []
    assert len(_SinkHandler.requests) == 1


def test_fresh_and_decided_shipments_never_fire(sink, monkeypatch):
    _enable(monkeypatch, sink)
    service = _service()
    service.analyze(dict(SHIPMENT))  # fresh: inside every budget
    assert service.sla_breach_sweep() == []

    _breaching(service, shipment_id="SLA-2")
    service.approve("SLA-2", approver="ops-lead")  # decided: no longer waiting
    assert service.sla_breach_sweep() == []
    assert _SinkHandler.requests == []


def test_reanalysis_resets_the_marker_and_can_fire_again(sink, monkeypatch):
    _enable(monkeypatch, sink)
    service = _service()
    _breaching(service)
    assert len(service.sla_breach_sweep()) == 1

    _breaching(service)  # a fresh analysis of the same shipment
    assert len(service.sla_breach_sweep()) == 1
    assert len(_SinkHandler.requests) == 2


def test_sweep_emits_a_stream_event(sink, monkeypatch):
    _enable(monkeypatch, sink)
    service = _service()
    _breaching(service)

    events = CollectingSink()
    service.sla_breach_sweep(event_sink=events)

    assert [e.type for e in events.events] == ["sla_breach"]
    assert events.events[0].detail["outcome"] == "sent"


# ---------------------------------------------------------------------------
# Opt-in discipline
# ---------------------------------------------------------------------------


def test_disabled_channel_observes_but_fires_nothing(sink, monkeypatch):
    monkeypatch.setenv("SLA_BREACH_WEBHOOK_URL", sink)  # URL set, channel off
    service = _service()
    record = _breaching(service)

    entries = service.sla_breach_sweep()
    assert entries[0]["outcome"] == "disabled"
    assert _SinkHandler.requests == []
    assert record.sla_breach_event_at is None  # unmarked: fires once enabled

    _enable(monkeypatch, sink)
    assert len(service.sla_breach_sweep()) == 1


def test_enabled_without_a_url_reports_not_configured(monkeypatch):
    monkeypatch.setenv("SLA_BREACH_WEBHOOK", "on")
    service = _service()
    record = _breaching(service)

    entries = service.sla_breach_sweep()
    assert entries[0]["outcome"] == "not_configured"
    assert record.sla_breach_event_at is None
    assert record.sla_dispatch_attempts == []


def test_breach_event_falls_back_to_the_approval_webhook_url(sink, monkeypatch):
    monkeypatch.setenv("SLA_BREACH_WEBHOOK", "on")
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)  # no SLA-specific URL
    service = _service()
    _breaching(service)

    assert service.sla_breach_sweep()[0]["outcome"] == "sent"
    assert json.loads(_SinkHandler.requests[0]["body"])["event"] == "sla_breach"


# ---------------------------------------------------------------------------
# Tenancy + persistence + API
# ---------------------------------------------------------------------------


def test_sweep_covers_every_tenant_and_scopes_on_request(sink, monkeypatch):
    _enable(monkeypatch, sink)
    service = _service()
    _breaching(service, shipment_id="SLA-1", tenant_id="acme")
    _breaching(service, shipment_id="SLA-1", tenant_id="globex")

    scoped = service.sla_breach_sweep(tenant_id="acme")
    assert [(e["shipment_id"], e["tenant_id"]) for e in scoped] == [
        ("SLA-1", "acme")
    ]
    rest = service.sla_breach_sweep()
    assert [(e["shipment_id"], e["tenant_id"]) for e in rest] == [
        ("SLA-1", "globex")
    ]
    bodies = [json.loads(r["body"]) for r in _SinkHandler.requests]
    assert [b["tenant_id"] for b in bodies] == ["acme", "globex"]


def test_sla_bookkeeping_survives_a_sqlite_round_trip(sink, monkeypatch, tmp_path):
    _enable(monkeypatch, sink)
    path = tmp_path / "approvals.db"
    service = _service(store=SQLiteStore(path))
    _breaching(service)
    service.sla_breach_sweep()

    reopened = SQLiteStore(path).get("SLA-1", tenant_id="default")
    assert reopened.sla_breach_event_at is not None
    assert reopened.sla_dispatch_attempts[0]["outcome"] == "sent"


def test_api_sweep_endpoint_fires_once(sink, monkeypatch):
    _enable(monkeypatch, sink)
    service = _service()
    _breaching(service)
    monkeypatch.setattr(api_module, "service", service)
    http = TestClient(api_module.app)

    first = http.post("/queue/sla-sweep")
    assert first.status_code == 200
    assert first.json()["count"] == 1
    assert first.json()["events"][0]["outcome"] == "sent"
    assert http.post("/queue/sla-sweep").json()["count"] == 0
