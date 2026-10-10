"""Per-severity escalation factors: a critical case climbs the
ladder faster than a low one.

The ladder's second rung was one global multiple of the budget.
Now ``QUEUE_SLA_ESCALATION_FACTOR_<SEVERITY>`` overrides the global
factor per severity (beside the per-severity budget overrides), the
queue item reports the factor that applied to it, and the
``sla_escalation`` payload names it. These tests pin the parsing
(per-severity wins, global is the fallback, every value clamps to
≥1.0), the projection, and the ladder end to end: two breaches
reported in rung-1 territory, of which only the critical one —
escalating at 1.5× its 4h budget — has blown its second rung when
the sweep returns, while the high case (3× its 24h budget) has not.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from shipment_agent.insights import (
    approval_queue,
    sla_escalation_factors_from_env,
)
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import load_sample_shipments
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

NOW = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)


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
    for var in (
        "ACTION_WEBHOOK_URL",
        "ACTION_WEBHOOK_SECRET",
        "SLA_BREACH_WEBHOOK",
        "SLA_BREACH_WEBHOOK_URL",
        "QUEUE_SLA_ESCALATION_FACTOR",
        "QUEUE_SLA_ESCALATION_FACTOR_CRITICAL",
        "QUEUE_SLA_ESCALATION_FACTOR_HIGH",
        "QUEUE_SLA_ESCALATION_FACTOR_MEDIUM",
        "QUEUE_SLA_ESCALATION_FACTOR_LOW",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.service.load_dotenv", lambda *a, **k: None)


def _service() -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=InMemoryStore(),
        checkpointer=False,
    )


def _sample(sample_id: str) -> dict:
    return next(
        s for s in load_sample_shipments() if s["shipment_id"] == sample_id
    )


def _age(service: ShipmentService, shipment_id: str, at: datetime, age_hours: float):
    store = service._get_store()
    record = store.get(shipment_id, tenant_id="default")
    record.created_at = (at - timedelta(hours=age_hours)).isoformat()
    store.save(record)
    return record


# ---------------------------------------------------------------------------
# Parsing: per-severity wins, global falls back, everything clamps
# ---------------------------------------------------------------------------


def test_factors_default_to_the_global_factor(monkeypatch):
    assert sla_escalation_factors_from_env() == {
        "critical": 2.0,
        "high": 2.0,
        "medium": 2.0,
        "low": 2.0,
    }
    monkeypatch.setenv("QUEUE_SLA_ESCALATION_FACTOR", "3")
    factors = sla_escalation_factors_from_env()
    assert factors["critical"] == 3.0 and factors["low"] == 3.0


def test_per_severity_override_wins_and_clamps(monkeypatch):
    monkeypatch.setenv("QUEUE_SLA_ESCALATION_FACTOR_CRITICAL", "1.5")
    monkeypatch.setenv("QUEUE_SLA_ESCALATION_FACTOR_LOW", "3")
    monkeypatch.setenv("QUEUE_SLA_ESCALATION_FACTOR_HIGH", "0.25")
    factors = sla_escalation_factors_from_env()
    assert factors["critical"] == 1.5
    assert factors["low"] == 3.0
    assert factors["medium"] == 2.0  # untouched: the global default
    assert factors["high"] == 1.0  # clamped: never before the breach


# ---------------------------------------------------------------------------
# The projection: each item carries its own factor
# ---------------------------------------------------------------------------


def test_queue_items_carry_their_severitys_factor(monkeypatch):
    monkeypatch.setenv("QUEUE_SLA_ESCALATION_FACTOR_CRITICAL", "1.5")
    monkeypatch.setenv("QUEUE_SLA_ESCALATION_FACTOR_HIGH", "3")
    service = _service()
    service.analyze(_sample("SYN-1006"))  # critical, budget 4h
    service.analyze(_sample("SYN-1001"))  # high, budget 24h
    _age(service, "SYN-1006", NOW, 5)
    _age(service, "SYN-1001", NOW, 30)
    records = service._get_store().records(tenant_id="default")
    items = {
        item["shipment_id"]: item
        for item in approval_queue(
            records,
            now=NOW,
            escalation_factor=sla_escalation_factors_from_env(),
        )
    }
    critical = items["SYN-1006"]
    assert critical["sla_escalation_factor"] == 1.5
    assert critical["sla_escalation_hours"] == 6.0  # 1.5 × 4h budget
    assert critical["sla_stage"] == "breach"  # 5h: past 4h, below 6h
    high = items["SYN-1001"]
    assert high["sla_escalation_factor"] == 3.0
    assert high["sla_escalation_hours"] == 72.0  # 3 × 24h budget
    assert high["sla_stage"] == "breach"


# ---------------------------------------------------------------------------
# The ladder honours the factors, end to end
# ---------------------------------------------------------------------------


def test_only_the_critical_case_escalates_under_its_faster_factor(
    sink, monkeypatch
):
    monkeypatch.setenv("SLA_BREACH_WEBHOOK", "on")
    monkeypatch.setenv("SLA_BREACH_WEBHOOK_URL", sink)
    monkeypatch.setenv("QUEUE_SLA_ESCALATION_FACTOR_CRITICAL", "1.5")
    monkeypatch.setenv("QUEUE_SLA_ESCALATION_FACTOR_HIGH", "3")
    service = _service()
    service.analyze(_sample("SYN-1006"))  # critical: budget 4h, rung 2 at 6h
    service.analyze(_sample("SYN-1001"))  # high: budget 24h, rung 2 at 72h

    # First sweep: both breaches fire in rung-1 territory.
    _age(service, "SYN-1006", NOW, 5)
    _age(service, "SYN-1001", NOW, 30)
    first = service.sla_breach_sweep(now=NOW)
    assert {(e["shipment_id"], e["rung"]) for e in first} == {
        ("SYN-1006", "breach"),
        ("SYN-1001", "breach"),
    }

    # Two hours later the critical case (7h) has blown its 6h rung;
    # the high case (32h) is nowhere near its 72h rung.
    later = NOW + timedelta(hours=2)
    second = service.sla_breach_sweep(now=later)
    assert [(e["shipment_id"], e["rung"], e["outcome"]) for e in second] == [
        ("SYN-1006", "escalation", "sent")
    ]
    body = json.loads(_SinkHandler.requests[-1]["body"])
    assert body["event"] == "sla_escalation"
    assert body["escalation_factor"] == 1.5
    assert body["escalation_threshold_hours"] == 6.0
