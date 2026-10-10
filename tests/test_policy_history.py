"""Corpus change events + document history: the audit trail for the
knowledge base.

Stored tenant documents are mutable rows; until this round an
operator who replaced an SOP left no trace of what changed, when,
or under whose key. Now every add / replace / remove lands an
append-only ledger entry beside the document (actor key id, time,
SHA-256 text hashes — never text), served at
``GET /policies/{id}/history``, and fires a signed
``corpus_changed`` webhook through the same receiver path as the
other four families. These tests pin the ledger on both hermetic
store backends, the override's prior hash (the bundled original's
text), tenant isolation of the history read, the API's 404
semantics, and the webhook's opt-in behaviour — off records
``disabled`` on the entry and sends nothing; on delivers a signed
body the receiver helpers name ``corpus_changed``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.policies_data import policies_for_tenant
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService, verify_webhook_body
from shipment_agent.store import InMemoryStore, SQLiteStore
from shipment_agent.webhooks import event_family, inspect_webhook

DOC = {
    "policy_id": "SOP-ACME-FORKLIFT",
    "title": "Forklift puncture quarantine (Acme)",
    "text": "When a forklift punctures shrink wrap, quarantine the pallet.",
}
DOC_V2 = {**DOC, "text": "When a forklift punctures shrink wrap, hold the pallet."}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _service(store=None) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store if store is not None else InMemoryStore(),
        checkpointer=False,
    )


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


def _lifecycle(service: ShipmentService, tenant="acme"):
    service.upsert_tenant_policy(
        tenant, DOC["policy_id"], DOC["title"], DOC["text"],
        actor_key_id=f"{tenant}:current",
    )
    service.upsert_tenant_policy(
        tenant, DOC["policy_id"], DOC_V2["title"], DOC_V2["text"],
        actor_key_id=f"{tenant}:previous",
    )
    assert service.remove_tenant_policy(
        tenant, DOC["policy_id"], actor_key_id=f"{tenant}:current"
    )
    return service.tenant_policy_history(tenant, DOC["policy_id"])


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_ledger_records_add_replace_remove_with_hashes(backend, tmp_path):
    store = (
        InMemoryStore()
        if backend == "memory"
        else SQLiteStore(tmp_path / "history.db")
    )
    history = _lifecycle(_service(store))
    assert [entry["action"] for entry in history] == ["add", "replace", "remove"]
    add, replace, remove = history
    assert add["prior_text_hash"] is None
    assert add["text_hash"] == _hash(DOC["text"])
    assert add["actor_key_id"] == "acme:current"
    assert replace["prior_text_hash"] == _hash(DOC["text"])
    assert replace["text_hash"] == _hash(DOC_V2["text"])
    assert replace["actor_key_id"] == "acme:previous"
    # The removal's trail survives the document: prior hash is the
    # removed text's, and there is no after-text.
    assert remove["prior_text_hash"] == _hash(DOC_V2["text"])
    assert remove["text_hash"] is None
    # Hashes and ids only — the document text itself is never ledgered.
    for entry in history:
        assert DOC["text"] not in json.dumps(entry)
        assert entry["webhook"]["outcome"] == "disabled"


def test_override_of_a_bundled_document_ledgers_the_bundled_prior():
    service = _service()
    bundled = next(
        p for p in policies_for_tenant("acme") if p["policy_id"] == "SOP-ACME-01"
    )
    service.upsert_tenant_policy(
        "acme", "SOP-ACME-01", "Revised reefer response", "Revised text."
    )
    (entry,) = service.tenant_policy_history("acme", "SOP-ACME-01")
    # The corpus served the bundled text before this write, so the
    # change is a replace whose prior hash is the bundled original's.
    assert entry["action"] == "replace"
    assert entry["prior_text_hash"] == _hash(bundled["text"])


def test_history_is_partitioned_by_tenant():
    service = _service()
    service.upsert_tenant_policy("acme", DOC["policy_id"], DOC["title"], DOC["text"])
    assert service.tenant_policy_history("globex", DOC["policy_id"]) == []
    assert len(service.tenant_policy_history("acme", DOC["policy_id"])) == 1


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_module, "service", _service())
    return TestClient(api_module.app)


def test_api_history_endpoint_and_404_semantics(client):
    headers = {"X-Tenant-ID": "acme"}
    missing = client.get(f"/policies/{DOC['policy_id']}/history", headers=headers)
    assert missing.status_code == 404
    client.post("/policies", json=DOC, headers=headers)
    client.delete(f"/policies/{DOC['policy_id']}", headers=headers)
    response = client.get(f"/policies/{DOC['policy_id']}/history", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "acme"
    assert [c["action"] for c in body["changes"]] == ["add", "remove"]
    # Another tenant's partition holds neither the document nor its
    # history: the same 404 as any cross-tenant read.
    other = client.get(
        f"/policies/{DOC['policy_id']}/history", headers={"X-Tenant-ID": "globex"}
    )
    assert other.status_code == 404


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


def test_corpus_changed_webhook_fires_signed_and_ledgers_its_outcome(
    sink, monkeypatch
):
    monkeypatch.setenv("SLA_BREACH_WEBHOOK", "on")
    monkeypatch.setenv("SLA_BREACH_WEBHOOK_URL", sink)
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", "history-secret")
    service = _service()
    service.upsert_tenant_policy("acme", DOC["policy_id"], DOC["title"], DOC["text"])

    (delivered,) = _SinkHandler.requests
    payload = json.loads(delivered["body"])
    assert event_family(payload) == "corpus_changed"
    assert payload["tenant_id"] == "acme"
    assert payload["action"] == "add"
    assert payload["text_hash"] == _hash(DOC["text"])
    assert DOC["text"] not in delivered["body"].decode("utf-8")
    # The receiver path verifies it: signature + family + event id.
    verdict = inspect_webhook(delivered["body"], delivered["signature"], "history-secret")
    assert verdict["verified"] is True
    assert verdict["family"] == "corpus_changed"
    assert verdict["event_id"].startswith("corpus_changed:")
    assert verify_webhook_body(
        delivered["body"], delivered["signature"], "history-secret"
    )
    (entry,) = service.tenant_policy_history("acme", DOC["policy_id"])
    assert entry["webhook"]["outcome"] == "sent"
    assert entry["webhook"]["http_status"] == 200


def test_webhook_on_but_unconfigured_records_not_configured(monkeypatch):
    monkeypatch.setenv("SLA_BREACH_WEBHOOK", "on")
    service = _service()
    service.upsert_tenant_policy("acme", DOC["policy_id"], DOC["title"], DOC["text"])
    (entry,) = service.tenant_policy_history("acme", DOC["policy_id"])
    assert entry["webhook"]["outcome"] == "not_configured"
