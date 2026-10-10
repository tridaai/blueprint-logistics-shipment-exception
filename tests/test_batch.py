"""Concurrent batch processing tests: service.analyze_batch + CLI --concurrency.

A batch runs the same pipeline per shipment, preserves input order,
captures one bad shipment's failure in its own item instead of failing
the batch, and is unchanged at concurrency 1. The store is shared by
the batch (items see each other as memory) with writes serialised by
the store itself.
"""

from __future__ import annotations

import copy

import pytest

from shipment_agent import cli
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import load_sample_shipments
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore, SQLiteStore


def _service(store=None) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=store or InMemoryStore(),
    )


def _samples() -> list[dict]:
    return [dict(s) for s in load_sample_shipments()]


def test_batch_of_all_samples_at_concurrency_4():
    service = _service()
    samples = _samples()
    items = service.analyze_batch(samples, concurrency=4)
    assert len(items) == 14
    assert [i.shipment_id for i in items] == [s["shipment_id"] for s in samples]
    assert all(i.error is None and i.result is not None for i in items)
    by_id = {i.shipment_id: i.result for i in items}
    assert by_id["SYN-1001"].classification.exception_type.value == "delay"
    assert by_id["SYN-1002"].classification.exception_type.value == "damage"
    assert by_id["SYN-1013"].classification.exception_type.value == "delay"
    assert by_id["SYN-1013"].validation.passed is False  # the guardrail trap sample
    assert by_id["SYN-1014"].classification.exception_type.value == "delay"
    assert by_id["SYN-1014"].injection_flags  # the adversarial sample is flagged
    # Every result stopped at the human gate; nothing took action.
    assert all(r.approval_status == "awaiting_approval" for r in by_id.values())
    assert all(r.external_action_taken is False for r in by_id.values())


def test_batch_at_concurrency_1_matches_one_by_one():
    samples = _samples()[:4]
    batch = _service().analyze_batch(samples, concurrency=1)
    single = _service()
    expected = [single.analyze(copy.deepcopy(s)) for s in samples]
    assert [i.result.classification.exception_type for i in batch] == [
        r.classification.exception_type for r in expected
    ]
    assert all(i.error is None for i in batch)


def test_one_bad_shipment_does_not_fail_the_batch():
    good = _samples()[0]
    bad = {"shipment_id": "BAD-1"}  # fails payload validation (no lane/dates)
    items = _service().analyze_batch([good, bad, _samples()[1]], concurrency=3)
    assert items[0].error is None and items[0].result is not None
    assert items[1].shipment_id == "BAD-1"
    assert items[1].result is None
    assert items[1].error  # the validation failure, captured per item
    assert items[2].error is None and items[2].result is not None


def test_batch_against_the_sqlite_store_serialises_writes(tmp_path):
    service = _service(SQLiteStore(tmp_path / "batch.db"))
    items = service.analyze_batch(_samples()[:5], concurrency=4)
    assert all(i.error is None for i in items)
    for item in items:
        stored = service.get(item.shipment_id)
        assert stored is not None
        assert stored.approval_status == "awaiting_approval"


@pytest.fixture
def clean_cli_env(monkeypatch):
    for var in (
        "MODEL_BACKEND", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "RETRIEVER",
        "RUN_TOKEN_BUDGET",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.model_backends.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.retriever.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.graph.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.reviewer.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.config.load_dotenv", lambda *a, **k: None)


def test_cli_all_with_concurrency_prints_a_batch_summary(clean_cli_env, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["shipment-agent", "--all", "--concurrency", "4"])
    cli.main()
    out = capsys.readouterr().out
    assert "Batch summary: 14 shipment(s) · concurrency 4" in out
    assert "wall-clock" in out
    assert "SYN-1001: delay — awaiting_approval" in out
    assert "SYN-1014: delay — awaiting_approval" in out
    headers = [line for line in out.splitlines() if line.startswith("Shipment SYN-")]
    assert len(headers) == 14  # every sample rendered


def test_cli_all_sequential_also_summarises(clean_cli_env, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["shipment-agent", "--all"])
    cli.main()
    out = capsys.readouterr().out
    assert "Batch summary: 14 shipment(s) · concurrency 1" in out
    headers = [line for line in out.splitlines() if line.startswith("Shipment SYN-")]
    assert len(headers) == 14
