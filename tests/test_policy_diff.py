"""Corpus change diffs + the corpus-wide recent-changes feed.

The change ledger (see test_policy_history.py) pinned *that* a
change happened — actor, time, the text's hash before and after.
This round's entries also say *what* changed: a line-diff summary
with exact counts and, on adds and replaces, the changed lines
themselves, capped so one entry can never carry a whole document.
A removal is the deliberate exception — counts and the final hash
only, so erased text stays erased. The same entries are also read
across documents at ``GET /policies/history`` (newest first), and
the bundled corpus still fabricates no history: it is code, and
its ledger begins at the first operator write.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import (
    DIFF_MAX_LINE_CHARS,
    DIFF_MAX_LINES,
    ShipmentService,
    build_corpus_changed_payload,
)
from shipment_agent.store import InMemoryStore, SQLiteStore

V1 = "Line one stays.\nLine two is replaced.\nLine three stays."
V2 = "Line one stays.\nLine two is rewritten.\nLine three stays.\nLine four is new."


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


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_module, "service", _service())
    return TestClient(api_module.app)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_diff_matches_the_actual_edit(backend, tmp_path):
    store = (
        InMemoryStore()
        if backend == "memory"
        else SQLiteStore(tmp_path / "diff.db")
    )
    service = _service(store)
    service.upsert_tenant_policy("acme", "SOP-DIFF", "Diff doc", V1)
    service.upsert_tenant_policy("acme", "SOP-DIFF", "Diff doc", V2)
    add, replace = service.tenant_policy_history("acme", "SOP-DIFF")

    add_diff = add["diff"]
    assert add_diff["lines_before"] == 0
    assert add_diff["lines_after"] == 3
    assert add_diff["added_count"] == 3
    assert add_diff["removed_count"] == 0
    assert add_diff["added"] == V1.splitlines()
    assert add_diff["removed"] == []
    assert add_diff["truncated"] is False

    diff = replace["diff"]
    assert diff["lines_before"] == 3
    assert diff["lines_after"] == 4
    # One line rewritten (one out, one in) and one appended.
    assert diff["removed_count"] == 1
    assert diff["added_count"] == 2
    assert diff["removed"] == ["Line two is replaced."]
    assert diff["added"] == ["Line two is rewritten.", "Line four is new."]
    # Unchanged lines are not "changed lines" on either side.
    assert "Line one stays." not in diff["added"]
    assert "Line three stays." not in diff["removed"]
    assert diff["truncated"] is False


def test_diff_caps_lines_and_characters():
    service = _service()
    before = "\n".join(f"old line {i}" for i in range(30))
    after = "\n".join(f"new line {i}" for i in range(30))
    service.upsert_tenant_policy("acme", "SOP-BIG", "Big", before)
    service.upsert_tenant_policy("acme", "SOP-BIG", "Big", after)
    (_, replace) = service.tenant_policy_history("acme", "SOP-BIG")
    diff = replace["diff"]
    # The counts are exact; only the stored lines are capped.
    assert diff["added_count"] == 30
    assert diff["removed_count"] == 30
    assert len(diff["added"]) == DIFF_MAX_LINES
    assert len(diff["removed"]) == DIFF_MAX_LINES
    assert diff["truncated"] is True

    service.upsert_tenant_policy("acme", "SOP-LONG", "Long", "x" * 500)
    service.upsert_tenant_policy("acme", "SOP-LONG", "Long", "y" * 500)
    (_, long_replace) = service.tenant_policy_history("acme", "SOP-LONG")
    assert len(long_replace["diff"]["added"][0]) == DIFF_MAX_LINE_CHARS
    assert long_replace["diff"]["truncated"] is True


def test_remove_records_counts_and_the_final_hash_but_no_lines():
    service = _service()
    text = "alpha\nbravo\ncharlie\ndelta"
    service.upsert_tenant_policy("acme", "SOP-GONE", "Gone", text)
    assert service.remove_tenant_policy("acme", "SOP-GONE")
    (_, remove) = service.tenant_policy_history("acme", "SOP-GONE")
    assert remove["action"] == "remove"
    assert remove["text_hash"] is None
    assert remove["prior_text_hash"] == _hash(text)
    diff = remove["diff"]
    assert diff["lines_before"] == 4
    assert diff["lines_after"] == 0
    assert diff["removed_count"] == 4
    assert diff["added"] == [] and diff["removed"] == []
    # Erased text stays erased: no line of the removed document
    # survives anywhere in its ledger entry.
    for line in text.splitlines():
        assert line not in json.dumps(remove)


def test_api_update_produces_a_diff_that_matches_the_edit(client):
    headers = {"X-Tenant-ID": "acme"}
    doc = {"policy_id": "SOP-API", "title": "Api doc", "text": V1}
    assert client.post("/policies", json=doc, headers=headers).status_code == 201
    assert (
        client.post("/policies", json={**doc, "text": V2}, headers=headers).status_code
        == 201
    )
    history = client.get("/policies/SOP-API/history", headers=headers).json()
    replace = history["changes"][-1]
    assert replace["action"] == "replace"
    assert replace["diff"]["removed"] == ["Line two is replaced."]
    assert replace["diff"]["added"] == [
        "Line two is rewritten.",
        "Line four is new.",
    ]


def test_feed_is_newest_first_across_documents_and_limited(client):
    headers = {"X-Tenant-ID": "acme"}
    client.post(
        "/policies",
        json={"policy_id": "SOP-A", "title": "A", "text": "a1"},
        headers=headers,
    )
    client.post(
        "/policies",
        json={"policy_id": "SOP-B", "title": "B", "text": "b1"},
        headers=headers,
    )
    client.post(
        "/policies",
        json={"policy_id": "SOP-A", "title": "A", "text": "a2"},
        headers=headers,
    )
    client.delete("/policies/SOP-B", headers=headers)

    feed = client.get("/policies/history", headers=headers).json()
    assert feed["tenant_id"] == "acme"
    assert feed["count"] == 4
    assert [(c["policy_id"], c["action"]) for c in feed["changes"]] == [
        ("SOP-B", "remove"),
        ("SOP-A", "replace"),
        ("SOP-B", "add"),
        ("SOP-A", "add"),
    ]
    # The feed carries the same entries the per-document read does.
    assert feed["changes"][1]["diff"]["added"] == ["a2"]

    limited = client.get("/policies/history?limit=2", headers=headers).json()
    assert [(c["policy_id"], c["action"]) for c in limited["changes"]] == [
        ("SOP-B", "remove"),
        ("SOP-A", "replace"),
    ]
    # Another tenant's feed is empty, not a view into this one.
    other = client.get("/policies/history", headers={"X-Tenant-ID": "globex"})
    assert other.json() == {"tenant_id": "globex", "count": 0, "changes": []}


def test_seeded_corpus_fabricates_no_history(client):
    service = api_module.service
    # The bundled corpus serves documents from the first moment…
    listing = client.get("/policies", headers={"X-Tenant-ID": "acme"}).json()
    bundled = next(d for d in listing if d["policy_id"] == "SOP-ACME-01")
    assert bundled["source"] == "bundled"
    # …but none of it is a recorded change: no entries, an honestly
    # empty feed, and the per-document 404 the endpoint promises
    # for a document this partition has never stored or changed.
    assert service.tenant_policy_changes("acme") == []
    assert service.tenant_policy_history("acme", "SOP-ACME-01") == []
    feed = client.get("/policies/history", headers={"X-Tenant-ID": "acme"}).json()
    assert feed["count"] == 0 and feed["changes"] == []
    missing = client.get(
        "/policies/SOP-ACME-01/history", headers={"X-Tenant-ID": "acme"}
    )
    assert missing.status_code == 404


def test_entries_recorded_before_diffs_carry_none(client):
    service = api_module.service
    legacy = {
        "policy_id": "SOP-LEGACY",
        "tenant_id": "acme",
        "action": "add",
        "actor_key_id": "acme:current",
        "at": "2026-09-01T00:00:00+00:00",
        "prior_text_hash": None,
        "text_hash": _hash("legacy text"),
        "webhook": {"outcome": "disabled", "http_status": None, "error": None},
    }
    service._get_store().record_tenant_policy_change("acme", legacy)
    (entry,) = service.tenant_policy_history("acme", "SOP-LEGACY")
    assert "diff" not in entry  # not recorded — never backfilled
    body = client.get(
        "/policies/SOP-LEGACY/history", headers={"X-Tenant-ID": "acme"}
    ).json()
    assert "diff" not in body["changes"][0]


def test_corpus_changed_payload_stays_hashes_only():
    service = _service()
    service.upsert_tenant_policy("acme", "SOP-WH", "Wh", V1)
    service.upsert_tenant_policy("acme", "SOP-WH", "Wh", V2)
    (_, replace) = service.tenant_policy_history("acme", "SOP-WH")
    payload = build_corpus_changed_payload(replace)
    # The ledger carries the diff; the event channel never does.
    assert "diff" not in payload
    assert "Line two is rewritten." not in json.dumps(payload)
    assert payload["prior_text_hash"] == _hash(V1)
    assert payload["text_hash"] == _hash(V2)
