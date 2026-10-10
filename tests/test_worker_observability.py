"""Worker observability: the background processes report themselves.

The dispatch-retry worker and the SLA sweep run as their own
processes; until now they reported totals only at exit, so a worker
that stopped sweeping looked exactly like a healthy one with
nothing to do. Every sweep now folds its outcome into a status row
in the store (last sweep time, sweep count, cumulative outcomes
overall and per tenant), and ``/metrics`` renders the rows as
Prometheus families beside the record aggregates. These tests pin
the store contract on both hermetic backends, the folding (sweeps
accumulate; a quiet sweep still heartbeats), both workers, and the
/metrics surface.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.metrics import compute_metrics, render_prometheus, worker_metrics
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore, SQLiteStore

SHIPMENT = {
    "shipment_id": "OBS-1",
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
        type(self).requests.append({"body": body})
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


# ---------------------------------------------------------------------------
# The store contract
# ---------------------------------------------------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryStore()
    return SQLiteStore(tmp_path / "approvals.db")


def test_worker_status_round_trip(store):
    assert store.worker_status("dispatch_retries") is None
    assert store.all_worker_status() == {}
    summary = {"worker": "dispatch_retries", "sweeps": 1, "updated_at": "t0"}
    store.save_worker_status("dispatch_retries", summary)
    assert store.worker_status("dispatch_retries") == summary
    store.save_worker_status("sla_sweep", {"worker": "sla_sweep", "sweeps": 3})
    assert set(store.all_worker_status()) == {"dispatch_retries", "sla_sweep"}
    # A later summary replaces the row wholesale.
    store.save_worker_status("dispatch_retries", {"worker": "dispatch_retries", "sweeps": 2})
    assert store.worker_status("dispatch_retries")["sweeps"] == 2


# ---------------------------------------------------------------------------
# The retry worker heartbeats
# ---------------------------------------------------------------------------


def test_retry_worker_records_each_sweep(sink, monkeypatch):
    _SinkHandler.failures_remaining = 1  # the approval dispatch fails
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    service = _service()
    service.analyze(dict(SHIPMENT))
    service.approve("OBS-1", approver="ops-lead")  # dispatch fails, retry due later

    future = datetime.now(timezone.utc) + timedelta(hours=1)
    totals = service.dispatch_retry_worker(
        threading.Event(), max_sweeps=1, now_fn=lambda: future
    )
    assert totals == {"sweeps": 1, "attempted": 1, "sent": 1, "failed": 0}

    status = service.worker_status()["dispatch_retries"]
    assert status["sweeps"] == 1
    assert status["totals"] == {"attempted": 1, "sent": 1, "failed": 0, "errors": 0}
    assert status["by_tenant"] == {"default": {"attempted": 1, "sent": 1}}
    assert status["last_sweep_at"] and status["last_sweep"]["attempted"] == 1

    # A second, quiet sweep still heartbeats — and changes no totals.
    service.dispatch_retry_worker(
        threading.Event(), max_sweeps=1, now_fn=lambda: future
    )
    status = service.worker_status()["dispatch_retries"]
    assert status["sweeps"] == 2
    assert status["totals"]["attempted"] == 1
    assert status["last_sweep"]["attempted"] == 0


def test_retry_entries_carry_their_tenant(sink, monkeypatch):
    _SinkHandler.failures_remaining = 1
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    service = _service()
    service.analyze(dict(SHIPMENT), tenant_id="acme")
    service.approve("OBS-1", approver="ops-lead", tenant_id="acme")

    future = datetime.now(timezone.utc) + timedelta(hours=1)
    entries = service.run_dispatch_retries_once(now=future)
    assert entries == [
        {"shipment_id": "OBS-1", "tenant_id": "acme", "outcome": "sent"}
    ]


# ---------------------------------------------------------------------------
# The SLA sweep heartbeats
# ---------------------------------------------------------------------------


def test_sla_sweep_records_its_observations():
    service = _service()
    service.analyze(dict(SHIPMENT))
    record = service._get_store().get("OBS-1", tenant_id="default")
    record.created_at = (
        datetime.now(timezone.utc) - timedelta(days=2)
    ).isoformat()
    service._get_store().save(record)

    entries = service.sla_breach_sweep()  # channel off: observed as disabled
    assert entries and entries[0]["outcome"] == "disabled"
    status = service.worker_status()["sla_sweep"]
    assert status["sweeps"] == 1
    assert status["totals"]["observed"] == 1
    assert status["totals"]["disabled"] == 1
    assert status["by_tenant"] == {"default": {"disabled": 1}}

    # A sweep over an empty queue still proves the worker ran.
    service.approve("OBS-1", approver="ops-lead")
    assert service.sla_breach_sweep() == []
    status = service.worker_status()["sla_sweep"]
    assert status["sweeps"] == 2
    assert status["last_sweep"]["observed"] == 0


def test_status_recording_never_breaks_a_sweep():
    """A store double without the status methods (the protocol grew;
    an injected double may predate it) must not fail the sweep."""

    class BareStore(InMemoryStore):
        save_worker_status = None
        worker_status = None
        all_worker_status = None

    service = _service(store=BareStore())
    assert service.sla_breach_sweep() == []
    assert service.worker_status() == {}


# ---------------------------------------------------------------------------
# The projection + the /metrics surface
# ---------------------------------------------------------------------------


def test_worker_metrics_projection():
    rows = {
        "dispatch_retries": {
            "worker": "dispatch_retries",
            "sweeps": 4,
            "last_sweep_at": "2026-10-10T14:00:00+00:00",
            "by_tenant": {"acme": {"attempted": 3, "sent": 2, "failed": 1}},
        }
    }
    projected = worker_metrics(rows)
    worker = projected["dispatch_retries"]
    assert worker["sweeps_total"] == 4
    assert worker["last_sweep_timestamp"] == pytest.approx(
        datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc).timestamp()
    )
    text = render_prometheus(compute_metrics([]), worker_status=rows)
    assert 'shipment_agent_worker_sweeps_total{worker="dispatch_retries"} 4' in text
    assert (
        'shipment_agent_worker_outcomes_total{worker="dispatch_retries",'
        'tenant="acme",outcome="sent"} 2'
    ) in text
    assert "shipment_agent_worker_last_sweep_timestamp" in text
    # Without rows, no worker families (the pre-round-6 shape).
    assert "worker_" not in render_prometheus(compute_metrics([]))


def test_metrics_endpoint_carries_the_worker_families(monkeypatch):
    service = _service()
    service.analyze(dict(SHIPMENT))
    service.sla_breach_sweep()  # records an sla_sweep row (nothing to observe)
    monkeypatch.setattr(api_module, "service", service)
    http = TestClient(api_module.app)

    text = http.get("/metrics").text
    assert 'shipment_agent_worker_sweeps_total{worker="sla_sweep"} 1' in text
    assert "shipment_agent_worker_last_sweep_timestamp" in text
    assert "shipment_agent_runs_total 1" in text  # the record families remain
