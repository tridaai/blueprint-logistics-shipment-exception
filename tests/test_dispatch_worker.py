"""The dispatch retry worker — the scheduler contract made real.

Round 3 recorded the backoff schedule on the delivery ledger and
left acting on it to the deployment. The worker loop is that actor:
it sweeps ``due_dispatch_retries`` and performs each due retry, in a
bounded, stoppable loop. These tests drive it with a fake clock (a
mutable cell the test advances) and a controllable sink, and pin
the bounds: max-sweeps stops the loop, a set stop event stops it
promptly, and one refused retry never stops a sweep.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from shipment_agent import cli as cli_module
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

SHIPMENT = {
    "shipment_id": "WRK-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


class _SinkHandler(BaseHTTPRequestHandler):
    """Fails the next N POSTs with HTTP 500, then accepts."""

    failures_remaining = 0

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
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
        "ACTION_WEBHOOK_MAX_ATTEMPTS",
        "ACTION_WEBHOOK_RETRY_BASE_SECONDS",
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


def _failed_delivery(service: ShipmentService, sink: str, monkeypatch, base: str = "0"):
    """Analyse + approve against a sink that fails the first POST."""
    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    monkeypatch.setenv("ACTION_WEBHOOK_RETRY_BASE_SECONDS", base)
    _SinkHandler.failures_remaining = 1
    service.analyze(dict(SHIPMENT))
    result = service.approve("WRK-1", approver="ops-lead")
    assert result.dispatch_status == "failed"
    return result


# ---------------------------------------------------------------------------
# One sweep
# ---------------------------------------------------------------------------


def test_one_sweep_performs_the_due_retry(sink, monkeypatch):
    service = _service()
    _failed_delivery(service, sink, monkeypatch)  # base 0: due at once

    outcomes = service.run_dispatch_retries_once()
    assert outcomes == [{"shipment_id": "WRK-1", "outcome": "sent"}]
    ledger = service.dispatch_ledger("WRK-1")
    assert ledger["attempts_used"] == 2
    assert ledger["dispatch_status"] == "sent"
    # Nothing left due: a second sweep is a no-op, not an error.
    assert service.run_dispatch_retries_once() == []


def test_sweep_respects_the_attempt_budget(sink, monkeypatch):
    service = _service()
    _failed_delivery(service, sink, monkeypatch, base="3600")
    # Pin the recorded due time into the past, then exhaust the
    # budget by capping attempts at the one already made: the sweep
    # must attempt nothing — the budget binds the worker exactly as
    # it binds a manual retry.
    record = service._get_store().get("WRK-1")
    record.dispatch_attempts[-1]["next_retry_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    ).isoformat()
    service._get_store().save(record)
    monkeypatch.setenv("ACTION_WEBHOOK_MAX_ATTEMPTS", "1")

    assert service.due_dispatch_retries() == []
    assert service.run_dispatch_retries_once() == []
    assert service.dispatch_ledger("WRK-1")["attempts_used"] == 1


# ---------------------------------------------------------------------------
# The worker loop
# ---------------------------------------------------------------------------


def test_worker_is_bounded_by_max_sweeps(sink, monkeypatch):
    service = _service()
    stop = threading.Event()
    started = time.perf_counter()
    totals = service.dispatch_retry_worker(
        stop, interval_seconds=0, max_sweeps=3
    )
    assert time.perf_counter() - started < 5  # bounded means bounded
    assert totals == {"sweeps": 3, "attempted": 0, "sent": 0, "failed": 0}
    assert not stop.is_set()  # the bound stopped it, not the event


def test_worker_acts_when_the_fake_clock_reaches_the_due_time(sink, monkeypatch):
    service = _service()
    _failed_delivery(service, sink, monkeypatch, base="3600")  # due in 1h

    clock = {"now": datetime.now(timezone.utc)}
    totals = service.dispatch_retry_worker(
        threading.Event(),
        interval_seconds=0,
        now_fn=lambda: clock["now"],
        max_sweeps=1,
    )
    assert totals["attempted"] == 0  # the backoff has not elapsed

    clock["now"] = clock["now"] + timedelta(hours=2)  # time passes
    totals = service.dispatch_retry_worker(
        threading.Event(),
        interval_seconds=0,
        now_fn=lambda: clock["now"],
        max_sweeps=1,
    )
    assert totals == {"sweeps": 1, "attempted": 1, "sent": 1, "failed": 0}
    assert service.dispatch_ledger("WRK-1")["dispatch_status"] == "sent"


def test_worker_stops_promptly_when_the_event_is_set(sink, monkeypatch):
    service = _service()
    stop = threading.Event()
    outcome: dict = {}

    def run():
        outcome["totals"] = service.dispatch_retry_worker(
            stop, interval_seconds=30  # long interval: only the event ends it
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(0.2)
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()  # the wait was on the event, not a sleep
    assert outcome["totals"]["sweeps"] >= 1


def test_worker_counts_a_delivery_that_fails_again(sink, monkeypatch):
    service = _service()
    _failed_delivery(service, sink, monkeypatch)  # base 0: due at once
    _SinkHandler.failures_remaining = 5  # the endpoint is still down
    totals = service.dispatch_retry_worker(
        threading.Event(), interval_seconds=0, max_sweeps=1
    )
    assert totals["attempted"] == 1
    assert totals["sent"] == 0
    assert totals["failed"] == 1
    assert service.dispatch_ledger("WRK-1")["attempts_used"] == 2


# ---------------------------------------------------------------------------
# The CLI shape
# ---------------------------------------------------------------------------


def test_cli_dispatch_retries_once(sink, monkeypatch, capsys, tmp_path):
    from shipment_agent.store import SQLiteStore

    monkeypatch.setenv("ACTION_WEBHOOK_URL", sink)
    monkeypatch.setenv("ACTION_WEBHOOK_RETRY_BASE_SECONDS", "0")
    service = ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=SQLiteStore(tmp_path / "state.db"),
        checkpointer=False,
    )
    _SinkHandler.failures_remaining = 1
    service.analyze(dict(SHIPMENT))
    service.approve("WRK-1", approver="ops-lead")
    assert service.dispatch_ledger("WRK-1")["dispatch_status"] == "failed"

    monkeypatch.setattr(cli_module, "build_service_from_env", lambda **kw: service)
    monkeypatch.setattr(
        "sys.argv", ["shipment-agent", "dispatch-retries", "--once"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()
    assert excinfo.value.code == 0
    assert "1 sent" in capsys.readouterr().out
    assert service.dispatch_ledger("WRK-1")["dispatch_status"] == "sent"


def test_cli_dispatch_retries_refuses_without_a_webhook(monkeypatch, capsys):
    monkeypatch.delenv("ACTION_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(
        "sys.argv", ["shipment-agent", "dispatch-retries", "--once"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()
    assert excinfo.value.code == 2
    assert "ACTION_WEBHOOK_URL" in capsys.readouterr().err
