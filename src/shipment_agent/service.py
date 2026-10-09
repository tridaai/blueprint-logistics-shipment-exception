"""Service layer: run + approval bookkeeping.

Approving a draft in this prototype records the decision and marks the claim
packet ready — it intentionally performs NO external action (no email, no
carrier API call). Production wiring replaces ``approve``'s final step with
the client's messaging / TMS integration, behind the same gate.

Records live in an approval store (``store.py``): SQLite on disk by
default, so analyses and human decisions survive restarts; the
in-memory store remains available as a test double.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import run_shipment
from .model_backends import ModelBackend, get_backend
from .retriever import Retriever, get_retriever
from .schemas import AgentResult, ShipmentInput
from .store import ApprovalRecord, ApprovalStore, default_store

__all__ = ["ApprovalRecord", "ShipmentService"]


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
        # Memory: what the store already knows about this consignee and
        # this lane becomes diagnosis evidence for the new analysis.
        history = self._history_summary(model)
        result = run_shipment(
            model, backend=backend, retriever=self.retriever, history=history
        )
        self._get_store().save(
            ApprovalRecord(result=result, shipment=model.model_dump(mode="json"))
        )
        return result

    def _history_summary(self, shipment: ShipmentInput) -> dict | None:
        """Summarise prior analysed shipments for this consignee + lane.

        Counts only priors that themselves had an exception, with the
        most recent exception types — the diagnosis cites them as
        evidence ("2 prior exceptions for this consignee in the stored
        history"). Returns None when there is nothing to remember.
        """
        priors = self._get_store().prior_shipments(
            exclude_shipment_id=shipment.shipment_id
        )
        if not priors:
            return None
        consignee = shipment.customer_name or next(
            (
                d.fields.get("consignee")
                for d in shipment.documents
                if d.fields.get("consignee")
            ),
            "",
        )
        lane = f"{shipment.origin} -> {shipment.destination}"

        def summarize(entries: list[dict]) -> tuple[int, list[str]]:
            hits = [e for e in entries if e["exception_type"] != "none"]
            return len(hits), [e["exception_type"] for e in hits[:3]]

        consignee_count, consignee_types = summarize(
            [e for e in priors if consignee and e["consignee"] == consignee]
        )
        lane_count, lane_types = summarize([e for e in priors if e["lane"] == lane])
        if not consignee_count and not lane_count:
            return None
        return {
            "consignee": consignee,
            "consignee_count": consignee_count,
            "consignee_recent_types": consignee_types,
            "lane": lane,
            "lane_count": lane_count,
            "lane_recent_types": lane_types,
        }

    def approve(self, shipment_id: str, approver: str) -> AgentResult:
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
        record.result.approval_status = "approved"
        record.result.decided_by = approver
        # NOTE: deliberately no external action here in the prototype.
        record.result.external_action_taken = False
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
        record.result.external_action_taken = False
        store.save(record)
        return record.result

    def get(self, shipment_id: str) -> AgentResult | None:
        record = self._get_store().get(shipment_id)
        return record.result if record else None
