"""Queue SLA views — age buckets and the breach flag.

A case that waits too long is itself an exception: every severity
has an age budget (defaults: critical 4h, high 24h, medium 48h, low
96h; QUEUE_SLA_HOURS_<SEVERITY> tunes it), and the queue item says
which bucket its age falls in and whether it has blown its budget.
These tests pin the bucket boundaries, the per-severity budgets, the
env override path the service uses, the summary the API reports,
and the console's breach rendering hooks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.insights import (
    DEFAULT_SLA_HOURS,
    age_bucket,
    approval_queue,
    queue_summary,
)
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import load_sample_shipments
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

BASE = datetime(2026, 10, 10, 8, 0, tzinfo=timezone.utc)


def _sample(sample_id: str, **overrides) -> dict:
    sample = next(
        s for s in load_sample_shipments() if s["shipment_id"] == sample_id
    )
    return {**sample, **overrides}


def _service() -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=InMemoryStore(),
        checkpointer=False,
    )


def _pin_age(service: ShipmentService, shipment_id: str, hours_ago: float) -> None:
    store = service._get_store()
    record = store.get(shipment_id)
    assert record is not None
    record.created_at = (BASE - timedelta(hours=hours_ago)).isoformat()
    store.save(record)


# ---------------------------------------------------------------------------
# Buckets and budgets (pure)
# ---------------------------------------------------------------------------


def test_age_bucket_boundaries():
    assert age_bucket(None) == "unknown"
    assert age_bucket(0) == "<1h"
    assert age_bucket(3599) == "<1h"
    assert age_bucket(3600) == "1-4h"
    assert age_bucket(4 * 3600 - 1) == "1-4h"
    assert age_bucket(4 * 3600) == "4-24h"
    assert age_bucket(24 * 3600 - 1) == "4-24h"
    assert age_bucket(24 * 3600) == "1-3d"
    assert age_bucket(72 * 3600 - 1) == "1-3d"
    assert age_bucket(72 * 3600) == ">3d"


def test_default_budgets_match_the_documented_values():
    assert DEFAULT_SLA_HOURS == {
        "critical": 4.0,
        "high": 24.0,
        "medium": 48.0,
        "low": 96.0,
    }


def _queue_at(service: ShipmentService, hours_after_base: float) -> list[dict]:
    return approval_queue(
        service._get_store().records(), now=BASE + timedelta(hours=hours_after_base)
    )


def test_breach_is_per_severity():
    service = _service()
    service.analyze(_sample("SYN-1006"))  # critical (delay)
    service.analyze(_sample("SYN-1001"))  # high (delay)
    service.analyze(_sample("SYN-1005"))  # low
    for sid in ("SYN-1006", "SYN-1001", "SYN-1005"):
        _pin_age(service, sid, 0)  # all analysed at BASE

    at_five_hours = {i["shipment_id"]: i for i in _queue_at(service, 5)}
    # Critical's 4h budget is blown by a 5h wait; high's 24h is not.
    assert at_five_hours["SYN-1006"]["sla_breach"] is True
    assert at_five_hours["SYN-1006"]["sla_hours"] == 4.0
    assert at_five_hours["SYN-1006"]["sla_overdue_seconds"] == 3600.0
    assert at_five_hours["SYN-1006"]["age_bucket"] == "4-24h"
    assert at_five_hours["SYN-1001"]["sla_breach"] is False
    assert at_five_hours["SYN-1001"]["sla_overdue_seconds"] == 0.0
    assert at_five_hours["SYN-1005"]["sla_breach"] is False

    at_five_days = {i["shipment_id"]: i for i in _queue_at(service, 120)}
    assert at_five_days["SYN-1005"]["sla_breach"] is True  # low budget: 96h
    assert at_five_days["SYN-1005"]["age_bucket"] == ">3d"


def test_explicit_thresholds_override_the_defaults():
    service = _service()
    service.analyze(_sample("SYN-1001"))  # high
    _pin_age(service, "SYN-1001", 0)
    items = approval_queue(
        service._get_store().records(),
        now=BASE + timedelta(hours=2),
        sla_hours={"high": 1.0},
    )
    (item,) = items
    assert item["sla_hours"] == 1.0
    assert item["sla_breach"] is True


def test_env_override_reaches_the_service_queue(monkeypatch):
    from shipment_agent.insights import sla_thresholds_from_env

    monkeypatch.setenv("QUEUE_SLA_HOURS_HIGH", "1")
    thresholds = sla_thresholds_from_env()
    assert thresholds["high"] == 1.0
    assert thresholds["critical"] == 4.0  # untouched severities keep defaults

    service = _service()
    service.analyze(_sample("SYN-1001"))  # high
    (item,) = service.approval_queue()
    assert item["sla_hours"] == 1.0  # the service used the env budget


# ---------------------------------------------------------------------------
# Summary + API
# ---------------------------------------------------------------------------


def test_queue_summary_counts():
    service = _service()
    service.analyze(_sample("SYN-1006"))  # critical
    service.analyze(_sample("SYN-1001"))  # high
    for sid in ("SYN-1006", "SYN-1001"):
        _pin_age(service, sid, 0)
    items = approval_queue(
        service._get_store().records(), now=BASE + timedelta(hours=30)
    )
    summary = queue_summary(items)
    assert summary["total"] == 2
    assert summary["sla_breaches"] == 2  # 30h blows both 4h and 24h
    assert summary["by_bucket"]["1-3d"] == 2
    assert summary["by_severity"] == {"critical": 1, "high": 1}
    assert summary["oldest_age_seconds"] == 30 * 3600


@pytest.fixture()
def sla_client(monkeypatch):
    service = _service()
    service.analyze(_sample("SYN-1001"))
    monkeypatch.setattr(api_module, "service", service)
    return TestClient(api_module.app)


def test_queue_endpoint_carries_the_sla_view(sla_client):
    body = sla_client.get("/queue").json()
    assert body["count"] == 1
    (item,) = body["queue"]
    assert item["age_bucket"] in ("<1h", "1-4h", "4-24h", "1-3d", ">3d")
    assert item["sla_hours"] == 24.0  # SYN-1001 is high severity
    assert item["sla_breach"] is False  # just analysed
    assert body["summary"]["total"] == 1
    assert body["summary"]["sla_breaches"] == 0
    assert body["sla_hours"] == {
        "critical": 4.0,
        "high": 24.0,
        "medium": 48.0,
        "low": 96.0,
    }


def test_console_renders_the_sla_column():
    html = (
        Path(__file__).parent.parent
        / "src"
        / "shipment_agent"
        / "static"
        / "index.html"
    ).read_text(encoding="utf-8")
    assert "<th>SLA</th>" in html
    assert "sla_breach" in html  # the breach badge + row highlight
    assert "sla_overdue_seconds" in html
    assert "age_bucket" in html
