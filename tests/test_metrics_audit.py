"""Metrics + audit export: aggregates and the decisions trail,
computed from the store (the system of record)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.metrics import compute_metrics, render_prometheus
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

SHIPMENT = {
    "shipment_id": "MET-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}


def _populated_service() -> ShipmentService:
    service = ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=InMemoryStore(),
        checkpointer=False,
    )
    service.analyze(SHIPMENT)
    service.approve("MET-1", approver="ops-lead", reason="checked the packet")
    service.analyze(dict(SHIPMENT, shipment_id="MET-2"))
    return service


def test_compute_metrics_aggregates_the_store():
    service = _populated_service()
    metrics = compute_metrics(service._get_store().records())
    assert metrics["runs_total"] == 2
    assert metrics["decisions"] == {"approved": 1, "rejected": 0, "awaiting": 1}
    assert metrics["guardrail_failed_runs"] == 0
    # Mock runs report telemetry (calls + latency) but no tokens/cost —
    # the totals must not pretend otherwise.
    assert metrics["telemetry_runs"] == 2
    assert metrics["input_tokens_total"] == 0
    assert metrics["cost_runs"] == 0


def test_render_prometheus_shape():
    metrics = compute_metrics(_populated_service()._get_store().records())
    text = render_prometheus(metrics)
    assert "shipment_agent_runs_total 2" in text
    assert 'shipment_agent_decisions_total{decision="approved"} 1' in text
    assert text.endswith("\n")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api_module, "service", _populated_service())
    return TestClient(api_module.app)


def test_metrics_endpoint_serves_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "shipment_agent_runs_total 2" in response.text


def test_audit_export_json(client):
    response = client.get("/audit/export")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    decided = next(r for r in body["decisions"] if r["shipment_id"] == "MET-1")
    assert decided["decision"] == "approved"
    assert decided["decided_by"] == "ops-lead"
    assert decided["decision_reason"] == "checked the packet"
    assert decided["exception_type"] == "delay"
    assert decided["created_at"] and decided["decided_at"]
    awaiting = next(r for r in body["decisions"] if r["shipment_id"] == "MET-2")
    assert awaiting["decision"] == "" and awaiting["decided_at"] == ""


def test_audit_export_csv(client):
    response = client.get("/audit/export", params={"format": "csv"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    lines = response.text.strip().splitlines()
    assert lines[0].startswith("shipment_id,exception_type,severity")
    assert any(line.startswith("MET-1,delay,") for line in lines)


def test_audit_export_rejects_unknown_format(client):
    response = client.get("/audit/export", params={"format": "xml"})
    assert response.status_code == 422
