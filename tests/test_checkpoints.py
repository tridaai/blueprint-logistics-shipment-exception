"""Checkpointed approval gate: graph state persists, decisions resume it.

With a checkpointer attached, an analysis pauses its graph at the
approval gate (thread_id = the shipment id at graph level; the
service namespaces it by tenant — ``<tenant>:<shipment_id>``) and
the service's approve/reject resumes the thread with the decision.
These tests pin
the contract: threads survive across service instances over the same
database files (a process restart), the gate semantics are unchanged
(a guardrail-failed draft still cannot be approved, and its thread
stays parked), re-analysis resets the thread, and CHECKPOINTS=off is
the old flow with no checkpoint file at all.
"""

from __future__ import annotations

import pytest

from shipment_agent.checkpoints import SqliteCheckpointSaver, get_checkpointer
from shipment_agent.graph import build_graph, resume_approval, run_shipment
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import load_sample_shipments, sample_shipment_models
from shipment_agent.config import DEFAULT_TENANT_ID
from shipment_agent.service import ShipmentService, checkpoint_thread_id
from shipment_agent.store import InMemoryStore


def _saver(tmp_path) -> SqliteCheckpointSaver:
    return SqliteCheckpointSaver(tmp_path / "checkpoints.db")


def _thread_snapshot(saver, thread_id):
    app = build_graph(checkpointer=saver)
    return app.get_state({"configurable": {"thread_id": thread_id}})


def _configure_env(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_DB_PATH", str(tmp_path / "state.db"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.db"))


# ---------------------------------------------------------------------------
# Graph level: pause at the gate, resume with the decision
# ---------------------------------------------------------------------------

def test_run_pauses_at_the_gate_and_resume_completes_it(tmp_path):
    saver = _saver(tmp_path)
    result = run_shipment(
        sample_shipment_models()[0],
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        checkpointer=saver,
    )
    assert result.approval_status == "awaiting_approval"
    assert result.external_action_taken is False

    snapshot = _thread_snapshot(saver, result.shipment_id)
    assert snapshot.next  # parked: nodes remain (the gate)
    assert snapshot.values["approval_status"] == "awaiting_approval"

    final = resume_approval(
        result.shipment_id, {"decision": "approved", "actor": "ops-lead"}, saver
    )
    assert final is not None
    assert final["approval_status"] == "approved"
    assert not _thread_snapshot(saver, result.shipment_id).next
    # A second resume finds nothing pending.
    assert resume_approval(result.shipment_id, {"decision": "approved"}, saver) is None


def test_saver_list_and_delete_thread(tmp_path):
    saver = _saver(tmp_path)
    result = run_shipment(
        sample_shipment_models()[0],
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": result.shipment_id}}
    assert saver.get_tuple(config) is not None
    assert len(list(saver.list(config))) >= 1
    saver.delete_thread(result.shipment_id)
    assert saver.get_tuple(config) is None
    assert resume_approval(result.shipment_id, {"decision": "approved"}, saver) is None


# ---------------------------------------------------------------------------
# Service level: decisions survive a restart (two instances, same files)
# ---------------------------------------------------------------------------

def test_approve_after_restart_resumes_the_thread(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    first = ShipmentService()
    result = first.analyze(sample_shipment_models()[0])
    assert result.approval_status == "awaiting_approval"

    # A new instance over the same database files = a process restart.
    second = ShipmentService()
    decided = second.approve(result.shipment_id, approver="ops-lead")
    assert decided.approval_status == "approved"
    assert decided.decided_by == "ops-lead"
    assert decided.external_action_taken is False
    assert second.get(result.shipment_id).approval_status == "approved"

    # The graph thread completed too: nothing left to resume.
    saver = get_checkpointer()
    assert saver is not None
    thread = checkpoint_thread_id(DEFAULT_TENANT_ID, result.shipment_id)
    assert not _thread_snapshot(saver, thread).next
    assert _thread_snapshot(saver, thread).values["approval_status"] == "approved"


def test_reject_after_restart_resumes_the_thread(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    first = ShipmentService()
    result = first.analyze(sample_shipment_models()[0])
    second = ShipmentService()
    decided = second.reject(result.shipment_id, reviewer="ops-lead", reason="wrong lane")
    assert decided.approval_status == "rejected"
    saver = get_checkpointer()
    thread = checkpoint_thread_id(DEFAULT_TENANT_ID, result.shipment_id)
    assert _thread_snapshot(saver, thread).values["approval_status"] == "rejected"


def test_guardrail_failed_draft_still_refused_and_thread_stays_parked(
    monkeypatch, tmp_path
):
    _configure_env(monkeypatch, tmp_path)
    samples = load_sample_shipments()
    syn1013 = next(s for s in samples if s["shipment_id"] == "SYN-1013")
    service = ShipmentService()
    result = service.analyze(syn1013)
    assert result.validation.passed is False
    with pytest.raises(ValueError, match="cannot be approved"):
        service.approve("SYN-1013", approver="ops-lead")
    # The refusal did not consume the thread: it is still parked, and a
    # reject can still land on it.
    saver = get_checkpointer()
    thread = checkpoint_thread_id(DEFAULT_TENANT_ID, "SYN-1013")
    assert _thread_snapshot(saver, thread).next
    decided = service.reject("SYN-1013", reviewer="ops-lead", reason="blocked draft")
    assert decided.approval_status == "rejected"
    assert not _thread_snapshot(saver, thread).next


def test_reanalysis_resets_the_thread(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    service = ShipmentService()
    shipment = sample_shipment_models()[0]
    service.analyze(shipment)
    service.approve(shipment.shipment_id, approver="ops-lead")
    # Re-analysing supersedes the decided run: fresh thread at the gate,
    # record back to awaiting, and a new decision can land.
    again = service.analyze(shipment)
    assert again.approval_status == "awaiting_approval"
    saver = get_checkpointer()
    thread = checkpoint_thread_id(DEFAULT_TENANT_ID, shipment.shipment_id)
    assert _thread_snapshot(saver, thread).next
    decided = service.approve(shipment.shipment_id, approver="ops-lead")
    assert decided.approval_status == "approved"


# ---------------------------------------------------------------------------
# CHECKPOINTS=off: the pre-existing flow, no checkpoint file
# ---------------------------------------------------------------------------

def test_checkpoints_off_uses_the_service_state_flow(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    monkeypatch.setenv("CHECKPOINTS", "off")
    assert get_checkpointer() is None
    service = ShipmentService(store=InMemoryStore())
    result = service.analyze(sample_shipment_models()[0])
    decided = service.approve(result.shipment_id, approver="ops-lead")
    assert decided.approval_status == "approved"
    assert not (tmp_path / "checkpoints.db").exists()
