"""The SLA escalation ladder: a breach that keeps aging pages again.

Round 5 made a breach fire once (``sla_breach``). This round adds
the ladder's second rung: a breach that was reported while still
in rung-1 territory (below ``QUEUE_SLA_ESCALATION_FACTOR`` × its
budget, default 2×) and then keeps aging past that threshold
re-fires as a signed ``sla_escalation`` event carrying the wait
duration — on its own ledger, deduped by its own marker, one rung
per record per sweep, in ladder order. A record first observed
already past both thresholds fires only its breach event (that
payload already carries the full wait), which is pinned here too.
The sink is the same fake HTTP stub the breach tests use.
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
from shipment_agent.insights import (
    approval_queue,
    queue_summary,
    sla_escalation_factor_from_env,
)
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore, SQLiteStore

SHIPMENT = {
    "shipment_id": "ESC-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "carrier": "Synthetic Carrier",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

NOW = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)
# The shipment classifies delay/high: budget 24h, escalation at 48h.


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
        "QUEUE_SLA_ESCALATION_FACTOR",
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


def _aged(service: ShipmentService, age_hours: float, shipment_id="ESC-1"):
    """Analyse a shipment and set its record's age, relative to NOW."""
    service.analyze(dict(SHIPMENT, shipment_id=shipment_id))
    store = service._get_store()
    record = store.get(shipment_id, tenant_id="default")
    record.created_at = (NOW - timedelta(hours=age_hours)).isoformat()
    store.save(record)
    return record


def _enable(monkeypatch, sink):
    monkeypatch.setenv("SLA_BREACH_WEBHOOK", "on")
    monkeypatch.setenv("SLA_BREACH_WEBHOOK_URL", sink)


# ---------------------------------------------------------------------------
# The projection: stages, thresholds, summary
# ---------------------------------------------------------------------------


def test_queue_items_carry_the_ladder_stage():
    service = _service()
    _aged(service, 1)
    _aged(service, 30, shipment_id="ESC-2")
    _aged(service, 50, shipment_id="ESC-3")
    records = service._get_store().records(tenant_id="default")

    items = {item["shipment_id"]: item for item in approval_queue(records, now=NOW)}
    assert items["ESC-1"]["sla_stage"] == "within_budget"
    assert items["ESC-1"]["severity"] == "high"  # budget 24h, escalation 48h

    assert items["ESC-2"]["sla_stage"] == "breach"
    assert items["ESC-2"]["sla_escalated"] is False
    assert items["ESC-2"]["sla_escalation_hours"] == 48.0

    assert items["ESC-3"]["sla_stage"] == "escalated"
    assert items["ESC-3"]["sla_escalated"] is True
    assert items["ESC-3"]["sla_escalation_overdue_seconds"] == pytest.approx(
        2 * 3600, abs=1
    )


def test_queue_summary_counts_escalations_beside_breaches():
    service = _service()
    _aged(service, 30)
    _aged(service, 50, shipment_id="ESC-2")
    items = approval_queue(service._get_store().records(), now=NOW)
    summary = queue_summary(items)
    assert summary["sla_breaches"] == 2  # the escalated item is also in breach
    assert summary["sla_escalations"] == 1


def test_escalation_factor_is_env_tunable_and_clamped(monkeypatch):
    assert sla_escalation_factor_from_env() == 2.0
    monkeypatch.setenv("QUEUE_SLA_ESCALATION_FACTOR", "3")
    assert sla_escalation_factor_from_env() == 3.0
    monkeypatch.setenv("QUEUE_SLA_ESCALATION_FACTOR", "0.5")
    assert sla_escalation_factor_from_env() == 1.0  # never before the breach


# ---------------------------------------------------------------------------
# The ladder, end to end
# ---------------------------------------------------------------------------


def test_breach_then_escalation_then_silence(sink, monkeypatch):
    _enable(monkeypatch, sink)
    service = _service()
    record = _aged(service, 30)  # past the 24h budget, below the 48h rung

    first = service.sla_breach_sweep(now=NOW)
    assert [(e["rung"], e["outcome"]) for e in first] == [("breach", "sent")]
    assert json.loads(_SinkHandler.requests[0]["body"])["event"] == "sla_breach"
    assert record.sla_breach_age_seconds == pytest.approx(30 * 3600, abs=1)

    record.created_at = (NOW - timedelta(hours=50)).isoformat()
    service._get_store().save(record)

    second = service.sla_breach_sweep(now=NOW)
    assert [(e["rung"], e["outcome"]) for e in second] == [("escalation", "sent")]
    body = json.loads(_SinkHandler.requests[1]["body"])
    assert body["event"] == "sla_escalation"
    assert body["wait_seconds"] == pytest.approx(50 * 3600, abs=1)
    assert body["age_seconds"] == body["wait_seconds"]
    assert body["breach_age_seconds"] == pytest.approx(30 * 3600, abs=1)
    assert body["escalation_threshold_hours"] == 48.0
    assert body["tenant_id"] == "default"

    # Each rung ledgered separately; the approval ledger untouched.
    assert len(record.sla_dispatch_attempts) == 1
    assert len(record.sla_escalation_attempts) == 1
    assert record.sla_escalation_attempts[0]["outcome"] == "sent"
    assert record.dispatch_attempts == []

    # Deduped per rung: nothing left to say.
    assert service.sla_breach_sweep(now=NOW) == []
    assert len(_SinkHandler.requests) == 2


def test_first_observed_past_both_thresholds_fires_the_breach_only(
    sink, monkeypatch
):
    """The breach payload already carries the full wait — a second
    event for the same observation would be noise, not a ladder."""
    _enable(monkeypatch, sink)
    service = _service()
    _aged(service, 50)

    entries = service.sla_breach_sweep(now=NOW)
    assert [(e["rung"], e["outcome"]) for e in entries] == [("breach", "sent")]
    assert json.loads(_SinkHandler.requests[0]["body"])["age_seconds"] == (
        pytest.approx(50 * 3600, abs=1)
    )
    assert service.sla_breach_sweep(now=NOW) == []
    assert len(_SinkHandler.requests) == 1


def test_disabled_escalation_is_observed_and_marks_nothing(sink, monkeypatch):
    _enable(monkeypatch, sink)
    service = _service()
    record = _aged(service, 30)
    assert service.sla_breach_sweep(now=NOW)[0]["rung"] == "breach"

    monkeypatch.delenv("SLA_BREACH_WEBHOOK")  # channel off at the second rung
    record.created_at = (NOW - timedelta(hours=50)).isoformat()
    service._get_store().save(record)
    entries = service.sla_breach_sweep(now=NOW)
    assert [(e["rung"], e["outcome"]) for e in entries] == [
        ("escalation", "disabled")
    ]
    assert record.sla_escalation_event_at is None

    _enable(monkeypatch, sink)
    assert service.sla_breach_sweep(now=NOW)[0]["rung"] == "escalation"


def test_failed_escalation_is_ledgered_and_still_fires_only_once(
    sink, monkeypatch
):
    _enable(monkeypatch, sink)
    service = _service()
    record = _aged(service, 30)
    service.sla_breach_sweep(now=NOW)

    _SinkHandler.failures_remaining = 5
    record.created_at = (NOW - timedelta(hours=50)).isoformat()
    service._get_store().save(record)
    entries = service.sla_breach_sweep(now=NOW)
    assert [(e["rung"], e["outcome"]) for e in entries] == [("escalation", "failed")]
    assert record.sla_escalation_attempts[0]["http_status"] == 500
    assert record.sla_escalation_event_at is not None  # an alarm, not a retry loop
    assert service.sla_breach_sweep(now=NOW) == []
    assert len(_SinkHandler.requests) == 2


def test_escalation_emits_a_stream_event(sink, monkeypatch):
    _enable(monkeypatch, sink)
    service = _service()
    record = _aged(service, 30)
    events = CollectingSink()
    service.sla_breach_sweep(now=NOW, event_sink=events)
    record.created_at = (NOW - timedelta(hours=50)).isoformat()
    service._get_store().save(record)
    service.sla_breach_sweep(now=NOW, event_sink=events)
    assert [e.type for e in events.events] == ["sla_breach", "sla_escalation"]
    assert events.events[1].detail["rung"] == "escalation"


def test_escalation_bookkeeping_survives_a_sqlite_round_trip(
    sink, monkeypatch, tmp_path
):
    _enable(monkeypatch, sink)
    path = tmp_path / "approvals.db"
    service = _service(store=SQLiteStore(path))
    _aged(service, 30)
    service.sla_breach_sweep(now=NOW)
    # The SQLite double hands out copies: re-read before re-aging,
    # or the save below would overwrite the sweep's own stamps.
    record = service._get_store().get("ESC-1", tenant_id="default")
    record.created_at = (NOW - timedelta(hours=50)).isoformat()
    service._get_store().save(record)
    service.sla_breach_sweep(now=NOW)

    reopened = SQLiteStore(path).get("ESC-1", tenant_id="default")
    assert reopened.sla_breach_age_seconds == pytest.approx(30 * 3600, abs=1)
    assert reopened.sla_escalation_event_at is not None
    assert reopened.sla_escalation_attempts[0]["outcome"] == "sent"


def test_api_queue_reports_the_stage_and_the_factor(monkeypatch):
    service = _service()
    monkeypatch.setattr(api_module, "service", service)
    service.analyze(dict(SHIPMENT))
    record = service._get_store().get("ESC-1", tenant_id="default")
    record.created_at = (
        datetime.now(timezone.utc) - timedelta(hours=50)
    ).isoformat()
    service._get_store().save(record)
    http = TestClient(api_module.app)

    body = http.get("/queue").json()
    item = body["queue"][0]
    assert item["sla_stage"] == "escalated"
    assert item["sla_escalated"] is True
    assert body["sla_escalation_factor"] == 2.0
    assert body["summary"]["sla_breaches"] == 1
    assert body["summary"]["sla_escalations"] == 1
