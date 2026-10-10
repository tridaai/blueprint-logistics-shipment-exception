"""Runtime posture: JSON log format, request IDs, readiness."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.observability import (
    JsonLogFormatter,
    new_request_id,
    request_id_ctx,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    # No DATABASE_URL in the suite environment: the app serves on
    # in-memory test doubles, which readiness must name honestly.
    return TestClient(api_module.app)


def test_json_log_formatter_shape():
    record = logging.LogRecord(
        name="shipment_agent.api",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request",
        args=(),
        exc_info=None,
    )
    record.method = "GET"
    record.path = "/health"
    record.status = 200
    record.duration_ms = 1.25
    token = request_id_ctx.set("req-123")
    try:
        payload = json.loads(JsonLogFormatter().format(record))
    finally:
        request_id_ctx.reset(token)
    assert payload["level"] == "INFO"
    assert payload["message"] == "request"
    assert payload["request_id"] == "req-123"
    assert payload["method"] == "GET"
    assert payload["status"] == 200
    assert "ts" in payload


def test_request_ids_are_unique():
    assert new_request_id() != new_request_id()


def test_readiness_names_the_test_double_mode(client: TestClient):
    response = client.get("/readiness")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert "test doubles" in body["database"]


def test_request_id_header_generated_and_echoed(client: TestClient):
    response = client.get("/health")
    assert response.headers.get("x-request-id")
    echoed = client.get("/health", headers={"X-Request-ID": "caller-id-42"})
    assert echoed.headers["x-request-id"] == "caller-id-42"
