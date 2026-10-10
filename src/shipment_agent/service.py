"""Service layer: run + approval bookkeeping.

Approving a draft records the decision and marks the claim packet
ready. By default it performs NO external action (no email, no carrier
API call). The one opt-in exception is output routing: when
``ACTION_WEBHOOK_URL`` is configured, the approved packet is POSTed to
that endpoint — the customer's system of choice — and the outcome is
recorded on the result (see ``dispatch_approval_webhook``).

Records live in an approval store (``store.py``): SQLite on disk by
default, so analyses and human decisions survive restarts; the
in-memory store remains available as a test double.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .config import env_float, env_str, load_dotenv
from .graph import run_shipment
from .model_backends import ModelBackend, get_backend
from .retriever import Retriever, get_retriever
from .schemas import AgentResult, ShipmentInput
from .store import ApprovalRecord, ApprovalStore, carrier_summary, default_store

__all__ = ["ApprovalRecord", "BatchItem", "ShipmentService"]


@dataclass
class BatchItem:
    """One outcome of a batch analysis: a result, or the error that
    stopped that one shipment — never the whole batch."""

    shipment_id: str
    result: AgentResult | None = None
    error: str | None = None


def dispatch_approval_webhook(record: ApprovalRecord) -> str | None:
    """POST the approved packet to ``ACTION_WEBHOOK_URL`` (off by default).

    This is the output-routing seam — the thin adapter between an
    approval and the customer's system of choice (their TMS, a ticket
    queue, an automation endpoint). Returns ``None`` when no URL is
    configured (the default: no external action at all), ``"sent"`` on
    a 2xx response, and ``"failed"`` on any error or non-2xx — a failed
    dispatch never undoes the approval; it is recorded on the result
    for follow-up. Short timeout on purpose: an approval must not hang
    on a downstream system.
    """
    load_dotenv()
    url = env_str("ACTION_WEBHOOK_URL")
    if not url:
        return None
    result = record.result
    payload = {
        "shipment_id": result.shipment_id,
        "approval_status": result.approval_status,
        "decided_by": result.decided_by,
        "classification": result.classification.model_dump(mode="json"),
        "draft": {"subject": result.draft.subject, "body": result.draft.body},
        "claim_packet": result.draft.claim_packet,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = env_float("ACTION_WEBHOOK_TIMEOUT_SECONDS", 5.0)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return "sent" if 200 <= response.status < 300 else "failed"
    except Exception:  # dispatch failure is recorded, never raised
        return "failed"


@dataclass
class ShipmentService:
    backend: ModelBackend | None = None
    retriever: Retriever | None = None
    store: ApprovalStore | None = None
    _resolved_store: ApprovalStore | None = field(default=None, repr=False)

    def _get_store(self) -> ApprovalStore:
        if self._resolved_store is None:
            self._resolved_store = self.store or default_store()
        return self._resolved_store

    def analyze(self, shipment: ShipmentInput | dict) -> AgentResult:
        # Backend and retriever come from the environment (MODEL_BACKEND /
        # RETRIEVER, with the repo-root .env loaded) unless injected.
        backend = self.backend or get_backend()
        if self.retriever is None:
            self.retriever = get_retriever()
        model = (
            shipment
            if isinstance(shipment, ShipmentInput)
            else ShipmentInput.model_validate(shipment)
        )
        return self._analyze_model(model, backend, self.retriever)

    def _analyze_model(
        self, model: ShipmentInput, backend: ModelBackend, retriever: Retriever
    ) -> AgentResult:
        # Memory: what the store already knows about this consignee and
        # this lane becomes diagnosis evidence for the new analysis.
        # The raw entries ride along too — the diagnosis tool loop
        # (provider mode) reads lane / carrier history through them.
        priors = self._get_store().prior_shipments(
            exclude_shipment_id=model.shipment_id
        )
        history = self._history_summary(model, priors)
        # Feedback loop: what human deciders said about earlier cases
        # for this consignee / lane (with their reasons) joins the same
        # evidence — the next diagnosis shows the pattern of oversight.
        feedback = self._feedback_for(model)
        if feedback:
            history = {**(history or {}), "feedback": feedback}
        result = run_shipment(
            model,
            backend=backend,
            retriever=retriever,
            history=history,
            priors=priors,
        )
        self._get_store().save(
            ApprovalRecord(result=result, shipment=model.model_dump(mode="json"))
        )
        return result

    def analyze_batch(
        self,
        shipments: list[ShipmentInput | dict],
        concurrency: int = 1,
    ) -> list[BatchItem]:
        """Analyse many shipments, in input order, optionally concurrently.

        Each item runs the same pipeline as :meth:`analyze` and gets
        its own result object. Injected backend/retriever are shared
        (test doubles are stateless); otherwise every item resolves
        fresh ones from the environment, so a provider backend's usage
        counters and a retriever's per-run stats never race across
        threads. The store is shared deliberately — batch items see
        each other as memory, like a real intake queue — and its
        writes are serialised by the store itself (SQLite: one
        connection per call; in-memory: a lock). A shipment that fails
        (bad payload, provider down) lands in its item's ``error``;
        it never fails the batch.
        """

        def run_one(raw: ShipmentInput | dict) -> BatchItem:
            shipment_id = (
                raw.get("shipment_id", "unknown")
                if isinstance(raw, dict)
                else getattr(raw, "shipment_id", "unknown")
            )
            try:
                model = (
                    raw
                    if isinstance(raw, ShipmentInput)
                    else ShipmentInput.model_validate(raw)
                )
                backend = self.backend or get_backend()
                retriever = self.retriever or get_retriever()
                result = self._analyze_model(model, backend, retriever)
                return BatchItem(shipment_id=model.shipment_id, result=result)
            except Exception as exc:  # one bad shipment never fails the batch
                return BatchItem(
                    shipment_id=str(shipment_id),
                    error=f"{type(exc).__name__}: {exc}",
                )

        workers = max(1, int(concurrency))
        if workers == 1 or len(shipments) <= 1:
            return [run_one(item) for item in shipments]
        semaphore = threading.Semaphore(workers)

        def guarded(item: ShipmentInput | dict) -> BatchItem:
            with semaphore:
                return run_one(item)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(guarded, shipments))

    @staticmethod
    def _consignee_of(shipment: ShipmentInput) -> str:
        """The consignee memory matches on: customer name, else a
        document's consignee field."""
        return shipment.customer_name or next(
            (
                d.fields.get("consignee")
                for d in shipment.documents
                if d.fields.get("consignee")
            ),
            "",
        )

    def _feedback_for(self, shipment: ShipmentInput) -> list[dict]:
        """Recent reviewer feedback matching this consignee or lane.

        Bounded to the last 3 matching decisions (the store returns
        newest first), reasons truncated — feedback is evidence, not a
        transcript. Each entry: {decision, reason, match} where match
        is "consignee" or "lane" (consignee wins when both match).
        """
        entries = self._get_store().decision_feedback(
            exclude_shipment_id=shipment.shipment_id
        )
        if not entries:
            return []
        consignee = self._consignee_of(shipment)
        lane = f"{shipment.origin} -> {shipment.destination}"
        matched: list[dict] = []
        for entry in entries:
            if consignee and entry["consignee"] == consignee:
                match = "consignee"
            elif entry["lane"] == lane:
                match = "lane"
            else:
                continue
            matched.append(
                {
                    "decision": entry["decision"],
                    "reason": entry["reason"][:160],
                    "match": match,
                }
            )
            if len(matched) == 3:
                break
        return matched

    def _history_summary(
        self, shipment: ShipmentInput, priors: list[dict] | None = None
    ) -> dict | None:
        """Summarise prior analysed shipments for this consignee + lane.

        Counts only priors that themselves had an exception, with the
        most recent exception types — the diagnosis cites them as
        evidence ("2 prior exceptions for this consignee in the stored
        history"). Returns None when there is nothing to remember.
        """
        if priors is None:
            priors = self._get_store().prior_shipments(
                exclude_shipment_id=shipment.shipment_id
            )
        if not priors:
            return None
        consignee = self._consignee_of(shipment)
        lane = f"{shipment.origin} -> {shipment.destination}"

        def summarize(entries: list[dict]) -> tuple[int, list[str]]:
            hits = [e for e in entries if e["exception_type"] != "none"]
            return len(hits), [e["exception_type"] for e in hits[:3]]

        consignee_count, consignee_types = summarize(
            [e for e in priors if consignee and e["consignee"] == consignee]
        )
        lane_count, lane_types = summarize([e for e in priors if e["lane"] == lane])
        carrier = carrier_summary(priors, shipment.carrier)
        if not consignee_count and not lane_count and not carrier["carrier_exception_count"]:
            return None
        return {
            "consignee": consignee,
            "consignee_count": consignee_count,
            "consignee_recent_types": consignee_types,
            "lane": lane,
            "lane_count": lane_count,
            "lane_recent_types": lane_types,
            "carrier": carrier["carrier"],
            "carrier_count": carrier["carrier_count"],
            "carrier_exception_count": carrier["carrier_exception_count"],
            "carrier_type_counts": carrier["carrier_type_counts"],
        }

    def approve(
        self, shipment_id: str, approver: str, reason: str = ""
    ) -> AgentResult:
        store = self._get_store()
        record = store.get(shipment_id)
        if record is None:
            raise KeyError(f"Unknown shipment_id: {shipment_id} (analyze it first)")
        if record.result.approval_status != "awaiting_approval":
            raise ValueError(
                f"Shipment {shipment_id} is not awaiting approval (status: {record.result.approval_status})"
            )
        if not record.result.validation.passed:
            raise ValueError(
                "Draft failed guardrail validation and cannot be approved: "
                + "; ".join(record.result.validation.errors)
            )
        record.approved = True
        record.approver = approver
        record.approve_reason = reason
        record.result.approval_status = "approved"
        record.result.decided_by = approver
        record.result.decision_reason = reason or None
        # Output routing: by default there is NO external action. When
        # the operator configures ACTION_WEBHOOK_URL, the approved
        # packet is POSTed to that endpoint (the customer's system of
        # choice) and the outcome is recorded — a failed dispatch leaves
        # the approval standing and says so on the result.
        record.result.external_action_taken = False
        dispatch = dispatch_approval_webhook(record)
        if dispatch is not None:
            record.dispatch_status = dispatch
            record.result.dispatch_status = dispatch
            if dispatch == "sent":
                record.result.external_action_taken = True
        store.save(record)
        return record.result

    def reject(self, shipment_id: str, reviewer: str, reason: str = "") -> AgentResult:
        store = self._get_store()
        record = store.get(shipment_id)
        if record is None:
            raise KeyError(f"Unknown shipment_id: {shipment_id} (analyze it first)")
        if record.result.approval_status != "awaiting_approval":
            raise ValueError(
                f"Shipment {shipment_id} is not awaiting approval (status: {record.result.approval_status})"
            )
        record.rejected_by = reviewer
        record.reject_reason = reason
        record.result.approval_status = "rejected"
        record.result.decided_by = reviewer
        record.result.decision_reason = reason or None
        record.result.external_action_taken = False
        store.save(record)
        return record.result

    def get(self, shipment_id: str) -> AgentResult | None:
        record = self._get_store().get(shipment_id)
        return record.result if record else None
