"""The receiver's half of the signed-webhook story.

Senders keep being re-implemented; receivers were too — and a
receiver that verifies wrong (over re-serialised JSON, without
dedupe, acking duplicates with errors) breaks the delivery contract
in ways the sender's ledger cannot diagnose. This file pins the
shipped receiver path end to end: the reference receiver in
``docs/webhook-receiver.py`` (stdlib only, imported here as a
module and served for real), the receiver-side helpers in
``webhooks.py``, and the ``verify-webhook`` CLI — over all four
event families through one code path.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from shipment_agent.service import sign_webhook_body, verify_webhook_body
from shipment_agent.webhooks import event_family, event_id_for, inspect_webhook

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET = "<redacted>"


def _load_receiver_module():
    spec = importlib.util.spec_from_file_location(
        "webhook_receiver", REPO_ROOT / "docs" / "webhook-receiver.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


receiver = _load_receiver_module()


# ---------------------------------------------------------------------------
# Payloads: one per family, built the way the sender builds them
# ---------------------------------------------------------------------------


def _approval_body() -> bytes:
    payload = {
        "shipment_id": "SYN-1001",
        "approval_status": "approved",
        "decided_by": "ops-lead",
        "classification": {"exception_type": "delay", "severity": "high"},
        "draft": {"subject": "Update on shipment SYN-1001", "body": "..."},
        "claim_packet": {"carrier": "Acme Freight"},
    }
    return json.dumps(payload).encode("utf-8")


def _record(shipment_id: str):
    """A real stored record for the payload builders to read."""
    from shipment_agent.model_backends import MockModelBackend
    from shipment_agent.retriever import KeywordRetriever
    from shipment_agent.service import ShipmentService
    from shipment_agent.store import InMemoryStore

    service = ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=InMemoryStore(),
        checkpointer=False,
    )
    payload = {
        "shipment_id": shipment_id,
        "origin": "Memphis, TN",
        "destination": "Charlotte, NC",
        "customer_name": "Synthetic Customer",
        "carrier": "Acme Freight",
        "scheduled_delivery": "2026-10-10T09:00:00",
        "estimated_delivery": "2026-10-11T21:00:00",
        "latest_event": "Delayed at regional hub due to weather hold",
        "documents": [],
    }
    service.analyze(payload)
    return service._get_store().get(shipment_id)


def _sla_breach_body() -> bytes:
    from shipment_agent.service import build_sla_breach_payload

    record = _record("SYN-2001")
    item = {
        "exception_type": "delay",
        "severity": "critical",
        "carrier": "Acme Freight",
        "lane": "Memphis, TN -> Charlotte, NC",
        "sla_hours": 4.0,
        "age_seconds": 20000.0,
        "sla_overdue_seconds": 5600.0,
    }
    payload = build_sla_breach_payload(record, item, "2026-10-10T12:00:00+00:00")
    return json.dumps(payload).encode("utf-8")


def _sla_escalation_body() -> bytes:
    from shipment_agent.service import build_sla_escalation_payload

    record = _record("SYN-2002")
    record.sla_breach_event_at = "2026-10-10T08:00:00+00:00"
    record.sla_breach_age_seconds = 15000.0
    item = {
        "exception_type": "delay",
        "severity": "critical",
        "carrier": "Acme Freight",
        "lane": "Memphis, TN -> Charlotte, NC",
        "sla_hours": 4.0,
        "sla_escalation_hours": 8.0,
        "sla_escalation_factor": 2.0,
        "age_seconds": 40000.0,
        "sla_escalation_overdue_seconds": 11200.0,
    }
    payload = build_sla_escalation_payload(record, item, "2026-10-10T12:00:00+00:00")
    return json.dumps(payload).encode("utf-8")


def _worker_stale_body() -> bytes:
    from shipment_agent.service import build_worker_stale_payload

    payload = build_worker_stale_payload(
        "sla_sweep",
        {
            "last_sweep_at": "2026-10-10T06:00:00+00:00",
            "age_seconds": 21600.0,
            "threshold_seconds": 3600.0,
        },
        "2026-10-10T12:00:00+00:00",
    )
    return json.dumps(payload).encode("utf-8")


FAMILY_BODIES = {
    "approval": _approval_body,
    "sla_breach": _sla_breach_body,
    "sla_escalation": _sla_escalation_body,
    "worker_stale": _worker_stale_body,
}


# ---------------------------------------------------------------------------
# The verification primitive + family / event-id helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(FAMILY_BODIES))
def test_verify_round_trips_for_every_family(family):
    body = FAMILY_BODIES[family]()
    signature = sign_webhook_body(body, SECRET)
    assert verify_webhook_body(body, signature, SECRET) is True
    verdict = inspect_webhook(body, signature, SECRET)
    assert verdict["verified"] is True
    assert verdict["family"] == family
    assert verdict["event_id"] == event_id_for(body, family)
    assert verdict["error"] is None


def test_verify_rejects_tampering_wrong_secret_and_missing_signature():
    body = _approval_body()
    signature = sign_webhook_body(body, SECRET)
    tampered = body.replace(b"approved", b"rejectd!")
    assert tampered != body
    assert verify_webhook_body(tampered, signature, SECRET) is False
    assert verify_webhook_body(body, signature, "a-different-secret") is False
    assert verify_webhook_body(body, None, SECRET) is False
    assert verify_webhook_body(body, "", SECRET) is False
    assert verify_webhook_body(body, signature, "") is False
    verdict = inspect_webhook(tampered, signature, SECRET)
    assert verdict["verified"] is False
    assert verdict["family"] is None  # never routed on a bad signature


def test_event_family_recognition():
    assert event_family({"event": "sla_breach"}) == "sla_breach"
    assert event_family({"event": "sla_escalation"}) == "sla_escalation"
    assert event_family({"event": "worker_stale"}) == "worker_stale"
    assert (
        event_family({"shipment_id": "X", "approval_status": "approved"})
        == "approval"
    )
    # An unknown event name is not quietly an approval, and a
    # shapeless body is nothing at all.
    assert event_family({"event": "something_else", "shipment_id": "X"}) is None
    assert event_family({"shipment_id": "X"}) is None
    assert event_family({}) is None


def test_event_id_is_stable_per_body_and_distinct_per_event():
    body = _sla_breach_body()
    assert event_id_for(body, "sla_breach") == event_id_for(body, "sla_breach")
    assert event_id_for(body, "sla_breach") != event_id_for(
        _worker_stale_body(), "worker_stale"
    )
    assert event_id_for(body, "sla_breach").startswith("sla_breach:")


# ---------------------------------------------------------------------------
# The reference receiver, served for real
# ---------------------------------------------------------------------------


@pytest.fixture
def receiver_server(tmp_path):
    log = receiver.EventLog(tmp_path / "events.jsonl")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), receiver.make_handler(SECRET, log)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", tmp_path / "events.jsonl"
    server.shutdown()


def _post(url: str, body: bytes, signature: str | None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if signature is not None:
        headers["X-Trida-Signature"] = signature
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.mark.parametrize("family", sorted(FAMILY_BODIES))
def test_receiver_accepts_each_family_once_and_dedupes_redelivery(
    receiver_server, family
):
    url, log_path = receiver_server
    body = FAMILY_BODIES[family]()
    signature = sign_webhook_body(body, SECRET)

    status, answer = _post(url + "/webhooks", body, signature)
    assert status == 200
    assert answer["received"] is True
    assert answer["family"] == family
    assert answer["duplicate"] is False
    assert answer["event_id"] == event_id_for(body, family)

    # A byte-identical redelivery: same ack, flagged duplicate —
    # the sender stops retrying, the effect applies once.
    status, answer = _post(url + "/webhooks", body, signature)
    assert status == 200
    assert answer["received"] is True
    assert answer["duplicate"] is True

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["duplicate"] is False
    assert json.loads(lines[1])["duplicate"] is True


def test_receiver_rejects_bad_signature_and_unsigned(receiver_server):
    url, log_path = receiver_server
    body = _approval_body()
    status, answer = _post(url, body, "sha256=" + "0" * 64)
    assert status == 401
    assert "signature" in answer["error"]
    status, _ = _post(url, body, None)
    assert status == 401
    assert not log_path.exists()  # nothing unauthenticated is recorded


def test_receiver_rejects_signed_garbage(receiver_server):
    url, _ = receiver_server
    body = b"this is not json"
    status, answer = _post(url, body, sign_webhook_body(body, SECRET))
    assert status == 400
    assert "JSON" in answer["error"]


def test_receiver_dedupe_survives_a_restart(tmp_path):
    log_path = tmp_path / "events.jsonl"
    body = _worker_stale_body()
    event_id = event_id_for(body, "worker_stale")
    first = receiver.EventLog(log_path)
    assert first.record(event_id, "worker_stale") is False
    assert first.record(event_id, "worker_stale") is True
    restarted = receiver.EventLog(log_path)
    assert restarted.record(event_id, "worker_stale") is True


# ---------------------------------------------------------------------------
# The verify-webhook CLI
# ---------------------------------------------------------------------------


def test_verify_webhook_cli(tmp_path, capsys, monkeypatch):
    from shipment_agent.cli import _run_verify_webhook

    body = _sla_escalation_body()
    payload_file = tmp_path / "payload.json"
    payload_file.write_bytes(body)
    signature = sign_webhook_body(body, SECRET)
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", SECRET)

    assert _run_verify_webhook([str(payload_file), "--signature", signature]) == 0
    out = capsys.readouterr().out
    assert "signature OK" in out
    assert "sla_escalation" in out
    assert event_id_for(body, "sla_escalation") in out

    assert _run_verify_webhook([str(payload_file), "--signature", "sha256=nope"]) == 1
    assert "FAILED" in capsys.readouterr().out

    headers_file = tmp_path / "headers.txt"
    headers_file.write_text(
        f"Content-Type: application/json\nX-Trida-Signature: {signature}\n",
        encoding="utf-8",
    )
    assert (
        _run_verify_webhook([str(payload_file), "--headers", str(headers_file)]) == 0
    )

    monkeypatch.delenv("ACTION_WEBHOOK_SECRET")
    assert _run_verify_webhook([str(payload_file), "--signature", signature]) == 2
