"""Worker staleness alerting: the observers get observed.

Worker status rows made a dead worker *visible* (a stale timestamp
on /metrics). This round makes the deployment *say* it: watched
workers (``WORKER_STALE_SECONDS_<WORKER>`` or the global
``WORKER_STALE_SECONDS``) whose last sweep is older than their
threshold — or who have never swept at all — show a stale flag on
/readiness and /metrics, and the SLA sweep fires one signed
``worker_stale`` event per staleness episode, ledgered on the
worker's own row and deduped by its episode marker until a fresh
sweep closes the episode. These tests pin the threshold parsing,
the projection, the episode lifecycle (fire once, silence, recover,
fire again), the channel-off posture, and both API surfaces.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.config import stale_watched_workers, worker_stale_seconds
from shipment_agent.metrics import worker_staleness
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

NOW = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)


class _SinkHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        type(self).requests.append(
            {"body": body, "signature": self.headers.get("X-Trida-Signature")}
        )
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
        "WORKER_STALE_SECONDS",
        "WORKER_STALE_SECONDS_DISPATCH_RETRIES",
        "WORKER_STALE_SECONDS_SLA_SWEEP",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.service.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.api.load_dotenv", lambda *a, **k: None)


def _service(store=None) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store or InMemoryStore(),
        checkpointer=False,
    )


def _row(store, worker: str, last_sweep_at: datetime) -> None:
    store.save_worker_status(
        worker,
        {
            "worker": worker,
            "sweeps": 3,
            "last_sweep_at": last_sweep_at.isoformat(),
            "updated_at": last_sweep_at.isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------


def test_thresholds_per_worker_over_global(monkeypatch):
    assert worker_stale_seconds("dispatch_retries") is None  # unwatched
    monkeypatch.setenv("WORKER_STALE_SECONDS", "600")
    assert worker_stale_seconds("dispatch_retries") == 600.0
    monkeypatch.setenv("WORKER_STALE_SECONDS_DISPATCH_RETRIES", "120")
    assert worker_stale_seconds("dispatch_retries") == 120.0
    assert worker_stale_seconds("sla_sweep") == 600.0  # global still applies
    assert stale_watched_workers() == ["dispatch_retries"]


def test_staleness_projection_ages_and_absence(monkeypatch):
    rows = {
        "dispatch_retries": {
            "last_sweep_at": (NOW - timedelta(seconds=900)).isoformat()
        },
        "sla_sweep": {"last_sweep_at": (NOW - timedelta(seconds=30)).isoformat()},
    }
    monkeypatch.setenv("WORKER_STALE_SECONDS", "300")
    view = worker_staleness(rows, now=NOW)
    assert view["dispatch_retries"]["stale"] is True
    assert view["dispatch_retries"]["age_seconds"] == 900.0
    assert view["sla_sweep"]["stale"] is False
    # Unwatched workers are absent from the view, never pronounced healthy.
    monkeypatch.delenv("WORKER_STALE_SECONDS")
    assert worker_staleness(rows, now=NOW) == {}


def test_a_watched_worker_that_never_swept_is_stale(monkeypatch):
    monkeypatch.setenv("WORKER_STALE_SECONDS_SLA_SWEEP", "300")
    view = worker_staleness({}, now=NOW)
    assert view["sla_sweep"]["stale"] is True
    assert view["sla_sweep"]["last_sweep_at"] is None
    assert view["sla_sweep"]["age_seconds"] is None


# ---------------------------------------------------------------------------
# The episode lifecycle: fire once, recover, fire again
# ---------------------------------------------------------------------------


def test_worker_stale_fires_once_per_episode(sink, monkeypatch):
    monkeypatch.setenv("SLA_BREACH_WEBHOOK", "on")
    monkeypatch.setenv("SLA_BREACH_WEBHOOK_URL", sink)
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", "stale-secret")
    monkeypatch.setenv("WORKER_STALE_SECONDS_DISPATCH_RETRIES", "300")
    service = _service()
    store = service._get_store()
    _row(store, "dispatch_retries", NOW - timedelta(seconds=900))

    fired = service.check_worker_staleness(now=NOW)
    assert [(e["worker"], e["outcome"]) for e in fired] == [
        ("dispatch_retries", "sent")
    ]
    body = json.loads(_SinkHandler.requests[0]["body"])
    assert body["event"] == "worker_stale"
    assert body["worker"] == "dispatch_retries"
    assert body["age_seconds"] == 900.0
    assert body["threshold_seconds"] == 300.0
    assert _SinkHandler.requests[0]["signature"].startswith("sha256=")

    # Ledgered on the worker's own row, episode marker set.
    row = store.worker_status("dispatch_retries")
    assert row["stale_episode"]["outcome"] == "sent"
    assert len(row["stale_alerts"]) == 1

    # A second check in the same episode fires nothing.
    assert service.check_worker_staleness(now=NOW) == []
    assert len(_SinkHandler.requests) == 1

    # The worker recovers: a fresh sweep closes the episode.
    service._record_worker_status("dispatch_retries", {"attempted": 0}, {})
    row = store.worker_status("dispatch_retries")
    assert "stale_episode" not in row
    assert len(row["stale_alerts"]) == 1  # the ledger is history, kept

    # It goes quiet again: a new episode alerts afresh.
    _row(store, "dispatch_retries", NOW - timedelta(seconds=900))
    fired = service.check_worker_staleness(now=NOW)
    assert [(e["worker"], e["outcome"]) for e in fired] == [
        ("dispatch_retries", "sent")
    ]
    assert len(_SinkHandler.requests) == 2


def test_channel_off_reports_but_marks_nothing(monkeypatch):
    monkeypatch.setenv("WORKER_STALE_SECONDS_DISPATCH_RETRIES", "300")
    service = _service()
    store = service._get_store()
    _row(store, "dispatch_retries", NOW - timedelta(seconds=900))
    fired = service.check_worker_staleness(now=NOW)
    assert [(e["worker"], e["outcome"]) for e in fired] == [
        ("dispatch_retries", "disabled")
    ]
    row = store.worker_status("dispatch_retries")
    assert "stale_episode" not in row  # enabling later still fires


def test_the_sla_sweep_runs_the_check(sink, monkeypatch):
    monkeypatch.setenv("SLA_BREACH_WEBHOOK", "on")
    monkeypatch.setenv("SLA_BREACH_WEBHOOK_URL", sink)
    monkeypatch.setenv("WORKER_STALE_SECONDS_DISPATCH_RETRIES", "300")
    service = _service()
    store = service._get_store()
    _row(store, "dispatch_retries", NOW - timedelta(seconds=900))
    service.sla_breach_sweep(now=NOW)
    bodies = [json.loads(r["body"]) for r in _SinkHandler.requests]
    assert [b["event"] for b in bodies] == ["worker_stale"]
    # The sweep's own fresh heartbeat is not judged stale.
    assert "sla_sweep" not in [
        e["worker"] for e in service.check_worker_staleness(now=NOW)
    ]


def test_never_swept_worker_alerts_with_null_age(sink, monkeypatch):
    monkeypatch.setenv("SLA_BREACH_WEBHOOK", "on")
    monkeypatch.setenv("SLA_BREACH_WEBHOOK_URL", sink)
    monkeypatch.setenv("WORKER_STALE_SECONDS_SLA_SWEEP", "300")
    service = _service()
    fired = service.check_worker_staleness(now=NOW)
    assert [(e["worker"], e["outcome"]) for e in fired] == [("sla_sweep", "sent")]
    body = json.loads(_SinkHandler.requests[0]["body"])
    assert body["last_sweep_at"] is None and body["age_seconds"] is None


# ---------------------------------------------------------------------------
# The API surfaces: /readiness flags it, /metrics renders it
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    service = _service()
    _row(
        service._get_store(),
        "dispatch_retries",
        datetime.now(timezone.utc) - timedelta(hours=2),
    )
    monkeypatch.setattr(api_module, "service", service)
    return TestClient(api_module.app)


def test_readiness_flags_the_stale_worker(client, monkeypatch):
    monkeypatch.setenv("WORKER_STALE_SECONDS_DISPATCH_RETRIES", "300")
    body = client.get("/readiness").json()
    assert body["status"] == "ready"  # degraded, not drained
    assert body["stale_workers"] == ["dispatch_retries"]
    watched = body["watched"]["dispatch_retries"]
    assert watched["stale"] is True
    assert watched["threshold_seconds"] == 300.0


def test_metrics_render_the_staleness_families(client, monkeypatch):
    monkeypatch.setenv("WORKER_STALE_SECONDS_DISPATCH_RETRIES", "300")
    text = client.get("/metrics").text
    assert 'shipment_agent_worker_stale{worker="dispatch_retries"} 1' in text
    assert "shipment_agent_worker_last_sweep_age_seconds" in text
