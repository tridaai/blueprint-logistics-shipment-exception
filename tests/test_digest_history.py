"""Digest history and the since-previous delta.

The digest used to be a single stored row — each sweep replaced
it, so "what moved since the last picture?" had no answer. The
sweep now also records one snapshot per tenant per digest in a
dated series (the corpus-ledger idiom: append-only, tenant-
partitioned, metadata and a content hash — a digest names ids
and stages, never shipment content), and
``GET /queue/digest?compare=previous`` diffs the newest snapshot
against its predecessor in code.

These tests pin the store series (both doubles, retention, the
tenant partition), the pure delta projection, and the two-sweep
story end to end: a shipment resolves, another climbs to the
escalated rung, a third arrives — and the delta names each move.
The SLA webhook sink is the same fake HTTP stub the breach-event
tests use; firings must be enabled for the ladder to mark and
ledger its rungs.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.insights import digest_delta
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import load_sample_shipments
from shipment_agent.service import ShipmentService
from shipment_agent.store import (
    DIGEST_SNAPSHOT_RETENTION,
    InMemoryStore,
    SQLiteStore,
)

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(hours=50)


class _SinkHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        type(self).requests.append({"ok": True})
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
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.service.load_dotenv", lambda *a, **k: None)


def _enable_sla(monkeypatch, sink_url: str) -> None:
    monkeypatch.setenv("SLA_BREACH_WEBHOOK", "on")
    monkeypatch.setenv("SLA_BREACH_WEBHOOK_URL", sink_url)


def _service(store=None) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store or InMemoryStore(),
        checkpointer=False,
    )


def _sample(shipment_id: str) -> dict:
    return next(
        s for s in load_sample_shipments() if s["shipment_id"] == shipment_id
    )


def _aged(service: ShipmentService, shipment_id: str, at: datetime, tenant_id=None):
    """Analyse a sample, then pin its record's creation time."""
    kwargs = {"tenant_id": tenant_id} if tenant_id else {}
    service.analyze(_sample(shipment_id), **kwargs)
    store = service._get_store()
    record = store.get(shipment_id, tenant_id=tenant_id or "default")
    record.created_at = at.isoformat()
    store.save(record)
    return record


# ---------------------------------------------------------------------------
# The store series
# ---------------------------------------------------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def history_store(request, tmp_path):
    if request.param == "memory":
        return InMemoryStore()
    return SQLiteStore(tmp_path / "approvals.db")


def _entry(tenant: str, n: int) -> dict:
    return {
        "tenant_id": tenant,
        "generated_at": f"2026-10-{10 + n:02d}T12:00:00+00:00",
        "awaiting": n,
        "stages": {"within_budget": 0, "breach": 0, "escalated": n},
        "items": {},
        "breaches_fired": [],
        "escalations_fired": [],
        "oldest_waiter_per_severity": {},
        "rotation": None,
        "content_hash": f"hash-{n}",
    }


def test_snapshot_series_round_trips_partitioned_by_tenant(history_store):
    history_store.record_digest_snapshot("acme", _entry("acme", 1))
    history_store.record_digest_snapshot("acme", _entry("acme", 2))
    history_store.record_digest_snapshot("globex", _entry("globex", 1))

    acme = history_store.digest_snapshots("acme")
    assert [e["awaiting"] for e in acme] == [1, 2]  # oldest first
    assert [e["awaiting"] for e in history_store.digest_snapshots("globex")] == [1]
    assert history_store.digest_snapshots("nobody") == []
    assert history_store.digest_snapshot_tenants() == ["acme", "globex"]
    # A limited read returns the newest entries, still oldest first.
    assert [
        e["awaiting"] for e in history_store.digest_snapshots("acme", limit=1)
    ] == [2]


def test_snapshot_series_prunes_to_the_retention_window(history_store):
    for n in range(DIGEST_SNAPSHOT_RETENTION + 5):
        history_store.record_digest_snapshot("acme", _entry("acme", n))
    series = history_store.digest_snapshots("acme")
    assert len(series) == DIGEST_SNAPSHOT_RETENTION
    assert series[0]["awaiting"] == 5  # the five oldest were pruned
    assert series[-1]["awaiting"] == DIGEST_SNAPSHOT_RETENTION + 4


# ---------------------------------------------------------------------------
# The delta projection (pure)
# ---------------------------------------------------------------------------


def _snapshot(items: dict, **overrides) -> dict:
    entry = {
        "tenant_id": "acme",
        "generated_at": "2026-10-10T12:00:00+00:00",
        "awaiting": len(items),
        "stages": {
            "within_budget": sum(1 for s in items.values() if s == "within_budget"),
            "breach": sum(1 for s in items.values() if s == "breach"),
            "escalated": sum(1 for s in items.values() if s == "escalated"),
        },
        "items": items,
        "breaches_fired": [],
        "escalations_fired": [],
        "oldest_waiter_per_severity": {},
        "rotation": None,
        "content_hash": "x",
    }
    entry.update(overrides)
    return entry


def test_digest_delta_names_each_kind_of_move():
    previous = _snapshot(
        {
            "SYN-A": "within_budget",  # -> breach: worsened
            "SYN-B": "breach",  # -> escalated: new escalation
            "SYN-C": "escalated",  # -> breach: improved
            "SYN-D": "breach",  # -> gone: resolved
            "SYN-E": "escalated",  # unchanged
        },
        breaches_fired=["SYN-B"],
        escalations_fired=["SYN-E"],
    )
    current = _snapshot(
        {
            "SYN-A": "breach",
            "SYN-B": "escalated",
            "SYN-C": "breach",
            "SYN-E": "escalated",
            "SYN-F": "within_budget",  # new, not escalated
            "SYN-G": "escalated",  # arrived already escalated
        },
        generated_at="2026-10-11T12:00:00+00:00",
        breaches_fired=["SYN-A", "SYN-G"],
        escalations_fired=["SYN-B", "SYN-G"],
    )
    delta = digest_delta(previous, current)
    assert delta["new_escalations"] == ["SYN-B", "SYN-G"]
    assert delta["worsened"] == ["SYN-A"]
    assert delta["improved"] == ["SYN-C"]
    assert delta["resolved"] == ["SYN-D"]
    assert delta["new_awaiting"] == ["SYN-F"]
    assert delta["breaches_fired_since"] == ["SYN-A", "SYN-G"]
    assert delta["escalations_fired_since"] == ["SYN-B", "SYN-G"]
    assert delta["awaiting"] == {"previous": 5, "current": 6}
    assert delta["stages"]["escalated"] == {"previous": 2, "current": 3}
    assert delta["since"] == "2026-10-10T12:00:00+00:00"
    assert delta["until"] == "2026-10-11T12:00:00+00:00"


def test_digest_delta_reads_the_rotation_window():
    opened = digest_delta(
        _snapshot({}), _snapshot({}, rotation={"grace_open": True})
    )
    assert opened["rotation"] == {
        "opened": True,
        "closed": False,
        "turned_ready_to_close": False,
    }
    ready = digest_delta(
        _snapshot({}, rotation={"grace_open": True, "ready_to_close": False}),
        _snapshot({}, rotation={"grace_open": True, "ready_to_close": True}),
    )
    assert ready["rotation"]["turned_ready_to_close"] is True
    assert ready["rotation"]["opened"] is False
    closed = digest_delta(
        _snapshot({}, rotation={"grace_open": True}), _snapshot({})
    )
    assert closed["rotation"]["closed"] is True


# ---------------------------------------------------------------------------
# The sweep leaves a series; the digest diffs it
# ---------------------------------------------------------------------------


def _two_sweep_story() -> ShipmentService:
    """Sweep at NOW, change the world, sweep again at LATER.

    At NOW: SYN-1001 (default, high) stands escalated at 100h old;
    SYN-1004 (acme, medium) stands in breach at 50h. Between the
    sweeps SYN-1001 is approved, and SYN-1002 (acme, high) arrives
    aged to stand in breach at LATER — where SYN-1004, now 100h
    old, has crossed its 96h escalation threshold. (The ladder
    climbs one rung per sweep: SYN-1004's breach fires in the
    first sweep, its escalation in the second.)
    """
    service = _service()
    _aged(service, "SYN-1001", NOW - timedelta(hours=100))
    _aged(service, "SYN-1004", NOW - timedelta(hours=50), tenant_id="acme")
    service.sla_breach_sweep(now=NOW)

    service.approve("SYN-1001", approver="op")
    _aged(service, "SYN-1002", LATER - timedelta(hours=30), tenant_id="acme")
    service.sla_breach_sweep(now=LATER)
    return service


def test_the_sweep_records_one_snapshot_per_tenant_per_digest(sink, monkeypatch):
    _enable_sla(monkeypatch, sink)
    service = _two_sweep_story()
    store = service._get_store()
    assert store.digest_snapshot_tenants() == ["acme", "default"]

    (first, second) = store.digest_snapshots("default")
    assert first["items"] == {"SYN-1001": "escalated"}
    assert first["awaiting"] == 1
    assert first["generated_at"] == NOW.isoformat()
    assert len(first["content_hash"]) == 64
    assert second["items"] == {}
    assert second["awaiting"] == 0

    (acme_first, acme_second) = store.digest_snapshots("acme")
    assert acme_first["items"] == {"SYN-1004": "breach"}
    assert acme_second["items"] == {
        "SYN-1002": "breach",
        "SYN-1004": "escalated",
    }
    # The series is partitioned: no tenant's snapshot names the
    # other's shipments.
    assert "SYN-1001" not in str(acme_first["items"])
    assert "SYN-1004" not in str(first["items"])


def test_compare_previous_names_what_moved(sink, monkeypatch):
    _enable_sla(monkeypatch, sink)
    service = _two_sweep_story()
    digest = service.queue_digest(compare=True)
    assert digest["stored"] is True
    delta = digest["delta"]

    default = delta["default"]
    assert default["available"] is True
    assert default["resolved"] == ["SYN-1001"]
    assert default["awaiting"] == {"previous": 1, "current": 0}
    assert default["stages"]["escalated"] == {"previous": 1, "current": 0}
    assert default["new_escalations"] == []

    acme = delta["acme"]
    assert acme["available"] is True
    assert acme["new_escalations"] == ["SYN-1004"]
    assert acme["new_awaiting"] == ["SYN-1002"]
    assert acme["escalations_fired_since"] == ["SYN-1004"]
    assert acme["awaiting"] == {"previous": 1, "current": 2}
    assert acme["since"] == NOW.isoformat()
    assert acme["until"] == LATER.isoformat()


def test_compare_without_a_previous_snapshot_says_so():
    service = _service()
    _aged(service, "SYN-1004", NOW - timedelta(hours=50), tenant_id="acme")
    digest = service.queue_digest(now=NOW, compare=True)
    assert digest["stored"] is False
    # Composed on read, with nothing stored before it: the delta
    # refuses to invent a baseline.
    assert digest["delta"]["acme"] == {
        "tenant_id": "acme",
        "available": False,
        "reason": "no earlier snapshot for this tenant",
    }
    # And the plain digest carries no delta section at all.
    assert "delta" not in service.queue_digest(now=NOW)


def test_compare_on_read_diffs_against_the_stored_snapshot():
    service = _service()
    _aged(service, "SYN-1004", NOW - timedelta(hours=50), tenant_id="acme")
    _aged(service, "SYN-1002", NOW - timedelta(hours=30), tenant_id="acme")
    store = service._get_store()
    store.record_digest_snapshot(
        "acme",
        {
            "tenant_id": "acme",
            "generated_at": (NOW - timedelta(hours=24)).isoformat(),
            "awaiting": 1,
            "stages": {"within_budget": 0, "breach": 1, "escalated": 0},
            "items": {"SYN-1004": "breach"},
            "breaches_fired": ["SYN-1004"],
            "escalations_fired": [],
            "oldest_waiter_per_severity": {},
            "rotation": None,
            "content_hash": "seeded",
        },
    )
    digest = service.queue_digest(now=NOW, compare=True)
    assert digest["stored"] is False  # composed on read, no summary row
    acme = digest["delta"]["acme"]
    assert acme["available"] is True
    assert acme["new_awaiting"] == ["SYN-1002"]
    assert acme["awaiting"] == {"previous": 1, "current": 2}
    assert acme["since"] == (NOW - timedelta(hours=24)).isoformat()


def test_digest_history_endpoint_serves_one_tenants_series(sink, monkeypatch):
    _enable_sla(monkeypatch, sink)
    service = _two_sweep_story()
    monkeypatch.setattr(api_module, "service", service)
    client = TestClient(api_module.app)

    body = client.get("/queue/digest/history", params={"tenant_id": "acme"}).json()
    assert body["tenant_id"] == "acme"
    assert len(body["snapshots"]) == 2
    assert body["snapshots"][0]["items"] == {"SYN-1004": "breach"}

    limited = client.get(
        "/queue/digest/history", params={"tenant_id": "acme", "limit": 1}
    ).json()
    assert len(limited["snapshots"]) == 1
    assert limited["snapshots"][0]["awaiting"] == 2

    compared = client.get("/queue/digest", params={"compare": "previous"}).json()
    assert compared["delta"]["acme"]["new_escalations"] == ["SYN-1004"]
    plain = client.get("/queue/digest").json()
    assert "delta" not in plain
