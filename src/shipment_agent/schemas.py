"""Typed schemas for the Shipment Exception Agent.

Everything crossing a node boundary is a Pydantic model (or its ``model_dump``),
so the graph state is always validated, serialisable data — never free-form
dicts invented mid-flight.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ExceptionType(str, Enum):
    DELAY = "delay"
    DAMAGE = "damage"
    DOCUMENT_MISMATCH = "document_mismatch"
    MISSED_APPOINTMENT = "missed_appointment"
    NONE = "none"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DocumentInput(BaseModel):
    """One shipment document (BOL, invoice, tracking note, …)."""

    doc_type: str = Field(description="bol | invoice | tracking | delivery_note | other")
    document_id: str
    raw_text: str = ""
    fields: dict[str, str] = Field(
        default_factory=dict,
        description="Structured fields extracted from the document, e.g. quantity_units, weight_kg, consignee.",
    )


class ShipmentInput(BaseModel):
    """A single shipment plus its latest known state. All sample data is synthetic."""

    shipment_id: str
    carrier: str = "Synthetic Carrier"
    origin: str
    destination: str
    customer_name: str = "Synthetic Customer"
    service_level: str = "standard"  # standard | priority | critical
    status: str = "in_transit"
    scheduled_delivery: datetime | None = None
    estimated_delivery: datetime | None = None
    latest_event: str = ""
    condition_notes: str = ""
    documents: list[DocumentInput] = Field(default_factory=list)


class Classification(BaseModel):
    exception_type: ExceptionType
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)
    rationale: str = ""


class ClassificationSuggestion(BaseModel):
    """Advisory LLM classification suggestion, recorded next to the rule result.

    Only ever produced when a real LLM backend is active and the rule
    result is low-confidence or ``none``. It never overrides the rule
    classification in ``AgentResult.classification`` — ``agrees_with_rules``
    flags disagreements for the human reviewer instead.
    """

    exception_type: str
    severity: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    backend: str = ""
    agrees_with_rules: bool = True


class RetrievedPolicy(BaseModel):
    policy_id: str
    title: str
    snippet: str
    score: float


class DocumentMismatch(BaseModel):
    field: str
    bol_value: str | None = None
    invoice_value: str | None = None


class DraftOutput(BaseModel):
    subject: str
    body: str
    claim_packet: dict[str, object] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)


class GuardrailCheck(BaseModel):
    """One named guardrail outcome — the demo/UI show these individually."""

    name: str
    passed: bool
    detail: str = ""
    blocking: bool = True


class ValidationResult(BaseModel):
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: list[GuardrailCheck] = Field(default_factory=list)


class TraceStep(BaseModel):
    """One pipeline step, with the actual output it produced.

    Additive field for the demo console: the UI renders these as the
    pipeline trace (ingest → classify → retrieve → draft → validate →
    human approval) so a reviewer can inspect what each node really did.
    """

    name: str  # ingest | classify | retrieve | draft | validate | human_approval
    title: str
    status: str = "completed"  # completed | passed | failed | awaiting
    summary: str = ""
    details: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    shipment_id: str
    classification: Classification
    llm_suggestion: ClassificationSuggestion | None = None
    delay_hours: float | None = None
    document_mismatches: list[DocumentMismatch] = Field(default_factory=list)
    policies: list[RetrievedPolicy] = Field(default_factory=list)
    draft: DraftOutput
    validation: ValidationResult
    trace: list[TraceStep] = Field(default_factory=list)
    approval_status: str = "awaiting_approval"
    external_action_taken: bool = False
    disclaimer: str = (
        "Reference prototype running on synthetic data. No external system "
        "was contacted and no customer message was sent. Draft requires "
        "human approval before any action."
    )
