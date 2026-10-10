"""PostgreSQL integration tests — gated on ``DATABASE_URL``.

The hermetic suite runs on in-memory doubles and never touches a
server. These tests exercise the production persistence path — the
Postgres approval store, the official LangGraph Postgres
checkpointer, and the migration runner — against a real database.
They skip cleanly when ``DATABASE_URL`` is not set.

Point ``DATABASE_URL`` at a *throwaway* database (the compose
stack's Postgres, or any pgvector-capable server): the tests run the
real migrations against it and write records with sample ids,
cleaning up the checkpoint threads they create.

Note on hermeticity: the suite-wide fixture scrubs ``DATABASE_URL``
from the environment before each test, so the URL is captured here
at import time (collection runs before fixtures) and re-applied per
test with ``monkeypatch``.
"""

from __future__ import annotations

import os

import pytest

_DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not _DATABASE_URL,
    reason="DATABASE_URL not set — Postgres integration tests skipped",
)

DELAY_SHIPMENT = {
    "shipment_id": "PG-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


@pytest.fixture()
def pg_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _DATABASE_URL)
    monkeypatch.delenv("STATE_DB_PATH", raising=False)
    return _DATABASE_URL


@pytest.fixture()
def clean_approvals(pg_env):
    """Start (and end) with no rows for the sample shipments."""
    from shipment_agent.db import connect, run_migrations

    run_migrations(pg_env)  # the approvals table must exist to clean it
    ids = ("PG-1", "PG-2")
    with connect(pg_env) as conn:
        conn.execute("DELETE FROM approvals WHERE shipment_id IN (%s, %s)", ids)
        conn.commit()
    yield ids
    with connect(pg_env) as conn:
        conn.execute("DELETE FROM approvals WHERE shipment_id IN (%s, %s)", ids)
        conn.commit()


def test_migrations_are_idempotent(pg_env):
    from shipment_agent.db import run_migrations

    first = run_migrations(pg_env)
    second = run_migrations(pg_env)
    assert second == []  # nothing left to apply the second time
    assert set(first) <= {"0001", "0002", "0003"}


def test_postgres_store_survives_reopen(clean_approvals):
    from shipment_agent.model_backends import MockModelBackend
    from shipment_agent.retriever import KeywordRetriever
    from shipment_agent.service import ShipmentService
    from shipment_agent.store import PostgresStore

    def service_on(store):
        return ShipmentService(
            backend=MockModelBackend(),
            retriever=KeywordRetriever(),
            store=store,
            checkpointer=False,
        )

    service_on(PostgresStore(_DATABASE_URL)).analyze(DELAY_SHIPMENT)
    service_on(PostgresStore(_DATABASE_URL)).approve(
        "PG-1", approver="ops-lead", reason="checked"
    )

    reopened = PostgresStore(_DATABASE_URL)
    record = reopened.get("PG-1")
    assert record is not None
    assert record.result.approval_status == "approved"
    assert record.approver == "ops-lead"
    assert record.approve_reason == "checked"
    assert record.decided_at  # the decision timestamp persisted
    assert record.created_at
    assert [r.result.shipment_id for r in reopened.records()] == ["PG-1"]


def test_postgres_checkpointer_pauses_and_resumes_across_instances(
    pg_env, clean_approvals
):
    """The gate's pause/resume over the official Postgres saver: one
    service instance pauses the thread, a fresh instance (its own
    service + saver resolution, same database) resumes it with the
    decision."""
    from shipment_agent.model_backends import MockModelBackend
    from shipment_agent.retriever import KeywordRetriever
    from shipment_agent.service import ShipmentService
    from shipment_agent.store import PostgresStore

    store = PostgresStore(_DATABASE_URL)  # the shared system of record

    def service():
        return ShipmentService(
            backend=MockModelBackend(),
            retriever=KeywordRetriever(),
            store=store,
            checkpointer=None,  # resolve from env -> the Postgres saver
        )

    shipment = dict(DELAY_SHIPMENT, shipment_id="PG-2")
    first = service()
    result = first.analyze(shipment)
    assert result.approval_status == "awaiting_approval"

    second = service()  # a "restart": new service, new saver instance
    try:
        decided = second.approve("PG-2", approver="ops-lead")
        assert decided.approval_status == "approved"
    finally:
        from shipment_agent.checkpoints import get_checkpointer

        saver = get_checkpointer()
        if saver is not None and hasattr(saver, "delete_thread"):
            saver.delete_thread("PG-2")
