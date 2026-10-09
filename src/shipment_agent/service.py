"""Service layer: run + approval bookkeeping.

Approving a draft in this prototype records the decision and marks the claim
packet ready — it intentionally performs NO external action (no email, no
carrier API call). Production wiring replaces ``approve``'s final step with
the client's messaging / TMS integration, behind the same gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import run_shipment
from .model_backends import ModelBackend, get_backend
from .retriever import Retriever, get_retriever
from .schemas import AgentResult, ShipmentInput


@dataclass
class ApprovalRecord:
    result: AgentResult
    approver: str | None = None
    approved: bool = False
    rejected_by: str | None = None
    reject_reason: str = ""


@dataclass
class ShipmentService:
    backend: ModelBackend | None = None
    retriever: Retriever | None = None
    _records: dict[str, ApprovalRecord] = field(default_factory=dict)

    def analyze(self, shipment: ShipmentInput | dict) -> AgentResult:
        # Backend and retriever come from the environment (MODEL_BACKEND /
        # RETRIEVER, with the repo-root .env loaded) unless injected.
        backend = self.backend or get_backend()
        if self.retriever is None:
            self.retriever = get_retriever()
        result = run_shipment(shipment, backend=backend, retriever=self.retriever)
        self._records[result.shipment_id] = ApprovalRecord(result=result)
        return result

    def approve(self, shipment_id: str, approver: str) -> AgentResult:
        record = self._records.get(shipment_id)
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
        # NOTE: deliberately no external action here in the prototype.
        record.result.external_action_taken = False
        return record.result

    def reject(self, shipment_id: str, reviewer: str, reason: str = "") -> AgentResult:
        record = self._records.get(shipment_id)
        if record is None:
            raise KeyError(f"Unknown shipment_id: {shipment_id} (analyze it first)")
        if record.result.approval_status != "awaiting_approval":
            raise ValueError(
                f"Shipment {shipment_id} is not awaiting approval (status: {record.result.approval_status})"
            )
        record.rejected_by = reviewer
        record.reject_reason = reason
        record.result.approval_status = "rejected"
        record.result.external_action_taken = False
        return record.result

    def get(self, shipment_id: str) -> AgentResult | None:
        record = self._records.get(shipment_id)
        return record.result if record else None
