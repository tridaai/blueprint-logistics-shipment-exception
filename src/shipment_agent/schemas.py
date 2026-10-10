"""Typed schemas for the Shipment Exception Agent.

Everything crossing a node boundary is a Pydantic model (or its ``model_dump``),
so the graph state is always validated, serialisable data — never free-form
dicts invented mid-flight.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


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


# The canonical document-type vocabulary. Callers (a TMS export, a
# carrier webhook) name documents in their own dialect — bill_of_lading,
# BOL, commercial_invoice, POD — so intake normalises onto this set.
CANONICAL_DOC_TYPES = ("bol", "invoice", "tracking", "delivery_note", "other")

_DOC_TYPE_ALIASES = {
    "bol": "bol",
    "bill_of_lading": "bol",
    "bill_of_lading_(bol)": "bol",
    "invoice": "invoice",
    "commercial_invoice": "invoice",
    "tracking": "tracking",
    "tracking_note": "tracking",
    "tracking_update": "tracking",
    "delivery_note": "delivery_note",
    "delivery_receipt": "delivery_note",
    "pod": "delivery_note",
    "proof_of_delivery": "delivery_note",
    "other": "other",
}


def normalize_doc_type(raw: str) -> tuple[str, bool]:
    """Map a caller-supplied doc_type onto the canonical vocabulary.

    Returns ``(canonical, flagged)``. Aliases and case/spelling variants
    resolve to the canonical type; an unknown type is kept as supplied
    (normalised for case and separators) but flagged, so downstream code
    — and the approver — can see it was not recognised.
    """
    key = re.sub(r"[\s\-]+", "_", str(raw).strip().lower())
    canonical = _DOC_TYPE_ALIASES.get(key, key)
    return canonical, canonical not in CANONICAL_DOC_TYPES


def _coerce_field_value(value: object) -> object:
    """Coerce a document field value to the string shape the pipeline uses.

    TMS/webhook payloads carry numbers as numbers (``120``,
    ``840.5``); the pipeline compares field values as strings, so intake
    coerces scalars instead of rejecting the payload. ``None`` values
    are dropped by the caller. Non-scalars pass through untouched and
    fail validation loudly, as they should.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    return value


class DocumentInput(BaseModel):
    """One shipment document (BOL, invoice, tracking note, …)."""

    doc_type: str = Field(
        description=(
            "Canonical: bol | invoice | tracking | delivery_note | other. "
            "Aliases (bill_of_lading, BOL, commercial_invoice, POD, …) are "
            "normalised at intake; unknown types are kept and flagged."
        )
    )
    doc_type_provided: str = Field(
        default="", description="The doc_type exactly as the caller supplied it."
    )
    doc_type_flagged: bool = Field(
        default=False,
        description="True when the supplied doc_type is outside the canonical vocabulary.",
    )
    document_id: str
    raw_text: str = ""
    fields: dict[str, str] = Field(
        default_factory=dict,
        description="Structured fields extracted from the document, e.g. quantity_units, weight_kg, consignee. Numeric values are coerced to strings at intake.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_intake(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        raw_type = data.get("doc_type")
        if isinstance(raw_type, str):
            canonical, flagged = normalize_doc_type(raw_type)
            data.setdefault("doc_type_provided", raw_type)
            data["doc_type"] = canonical
            data["doc_type_flagged"] = flagged
        fields = data.get("fields")
        if isinstance(fields, dict):
            data["fields"] = {
                key: _coerce_field_value(value)
                for key, value in fields.items()
                if value is not None
            }
        return data


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
    """The LLM half of the classification cross-check, in its original shape.

    Kept for API compatibility: ``AgentResult.cross_check`` carries the
    full cross-check record (both results + the resolution). This field
    mirrors the LLM classification itself; ``agrees_with_rules`` compares
    it against the rule result. In v2 the resolution policy in
    ``crosscheck.py`` decides which result ``AgentResult.classification``
    carries — an LLM result is adopted only when the rules landed on
    ``none``/low confidence and the LLM was highly confident, and that
    adoption is always flagged.
    """

    exception_type: str
    severity: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    backend: str = ""
    agrees_with_rules: bool = True


class FieldExtraction(BaseModel):
    """One document field as extracted, cross-checked against the provided value.

    ``confidence`` is the model's per-field confidence in provider mode;
    it is ``None`` in the default mode, where fields are source-provided
    (``status == "source_provided"``) and no extraction ran.
    """

    field: str
    provided_value: str | None = None
    extracted_value: str | None = None
    confidence: float | None = None
    status: str  # match | mismatch | missing_in_text | extracted_only | source_provided


class DocumentExtraction(BaseModel):
    """Extraction result for one document, with the deterministic cross-check."""

    document_id: str
    doc_type: str
    source: str  # llm | provided
    fields: list[FieldExtraction] = Field(default_factory=list)
    note: str = ""  # set when the LLM path failed and provided fields were used


class ClassificationCrossCheck(BaseModel):
    """Rules-vs-LLM classification cross-check and its resolution.

    The deterministic rules and (in provider mode) the LLM classify
    independently. ``resolution`` records what happened:

    - ``rules_only`` — no LLM classification was available (default mode,
      or the provider call failed); the rule result stands alone.
    - ``agree`` — both paths landed on the same exception type.
    - ``rules_authoritative`` — disagreement; the rule result stands.
    - ``llm_adopted`` — disagreement, but the rules landed on ``none`` or
      low confidence while the LLM was highly confident, so the LLM
      result was adopted — always flagged for the approver.
    """

    rule_exception_type: str
    rule_severity: str
    rule_confidence: float
    llm_exception_type: str | None = None
    llm_severity: str | None = None
    llm_confidence: float | None = None
    llm_backend: str = ""
    agrees: bool | None = None
    resolution: str  # rules_only | agree | rules_authoritative | llm_adopted
    adopted_source: str  # rules | llm
    note: str = ""


class Diagnosis(BaseModel):
    """Root-cause note over the computed evidence, with citations."""

    root_cause: str
    summary: str
    evidence: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    source: str  # template | llm
    note: str = ""  # set when the LLM composition failed and the template was used
    # Agentic diagnosis (provider mode): the tool calls the model made
    # while diagnosing, each {"name", "summary"} — the trace shows them.
    # Empty in the default mode, where no tool loop runs.
    tool_calls: list[dict] = Field(default_factory=list)


class RecoveryOption(BaseModel):
    """One proposed recovery option with deterministic impact scores.

    The proposer (template in default mode, LLM in provider mode) only
    names the option. Every number here — ETA improvement, added cost,
    SLA score, and the total ``score`` — is computed by deterministic
    code in ``options.py``; a model never does this arithmetic.
    """

    option_id: str
    kind: str
    title: str
    description: str
    eta_improvement_hours: float
    added_cost_units: float
    sla_score: float
    score: float
    recommended: bool = False


class RetrievedPolicy(BaseModel):
    policy_id: str
    title: str
    snippet: str
    score: float
    retrieval: str = ""  # keyword | semantic | keyword+semantic (hybrid merge info)


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


class VerificationResult(BaseModel):
    """The agent's self-critique of its own draft, before guardrails run.

    Two producers, one shape: an LLM critique in provider mode
    (``source="llm"``), or a deterministic evidence checklist in the
    default/mock mode (``source="checklist"``) — labelled honestly,
    never presented as model judgement when it is code.
    """

    grounded: bool
    issues: list[str] = Field(default_factory=list)
    source: str  # llm | checklist
    summary: str = ""
    note: str = ""  # set when the LLM critique failed and the checklist ran


class ValidationResult(BaseModel):
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: list[GuardrailCheck] = Field(default_factory=list)


class ReviewerResult(BaseModel):
    """The independent reviewer's verdict on the draft (see ``reviewer.py``).

    A generator/critic split: the reviewer did not draft the update and
    reads it adversarially — grounding, policy compliance, tone, and
    whether the approver has everything they need. Two producers, one
    shape: an LLM reviewer in provider mode (``source="llm"``, possibly
    on its own model — ``REVIEWER_MODEL``), or a deterministic second
    checklist in the default/mock mode (``source="checklist"``) running
    *different* checks than self-verification. A ``block`` verdict never
    auto-rejects: it flags the result (``reviewer_blocked``) and forces
    the autonomy recommendation to ineligible, and the human decides.
    """

    verdict: str  # pass | concerns | block
    findings: list[str] = Field(default_factory=list)
    source: str  # llm | checklist
    model: str | None = None
    note: str = ""  # set when the LLM review failed and the checklist ran


class InjectionFlag(BaseModel):
    """One prompt-injection screen hit on an untrusted input field.

    The screen (``screening.py``) runs at ingest over the fields a
    carrier or terminal system controls — latest event, condition
    notes, document text — and flags instruction-like content aimed at
    the agent ("ignore your policies", "approve this claim", an
    impersonated system message). Flags are evidence for the approver;
    the flagged sentences are additionally kept out of drafts and
    prompts by sanitisation at the construction sites.
    """

    field: str  # latest_event | condition_notes | document <id> raw_text
    pattern: str  # system_impersonation | ignore_instructions | policy_override | directed_approval | directed_promise
    excerpt: str  # the offending sentence (truncated)


class InformationRequest(BaseModel):
    """A composed clarification request for an under-determined case.

    Built when the classification is ``none`` at low confidence AND
    concrete inputs are missing (see ``clarify.py``). ``missing_items``
    is computed by code; ``message`` is the composed request text
    (LLM in provider mode, template in the default mode). The request
    is attached to the result and the claim packet — never sent by
    the pipeline.
    """

    message: str
    missing_items: list[str] = Field(default_factory=list)
    source: str  # llm | template


class TokenBudget(BaseModel):
    """``RUN_TOKEN_BUDGET`` accounting for one run (a cost guardrail).

    Set via the environment; unset means off and this object is absent.
    When the run's cumulative provider tokens pass the limit, the
    remaining provider steps degrade to their deterministic/template
    paths (with trace notes) — a run is never hard-failed for budget.
    """

    limit: int
    used: int
    exceeded: bool


class RunTelemetry(BaseModel):
    """Per-run observability: what the run used and how long it took.

    In provider mode the token counts are the provider's own reported
    usage, aggregated across every model call in the run (extraction,
    cross-check, diagnosis loop, options, draft, verification, repair),
    and ``estimated_cost_usd`` applies the small price table in
    ``model_backends.py`` — an estimate, labelled as such; an unlisted
    model reports None. In the default (mock) mode no model ran, so
    tokens and cost are None — never fabricated — while the call count
    (template renders) and the wall-clock latency stay real. ``budget``
    is present only when ``RUN_TOKEN_BUDGET`` is set.
    """

    backend: str
    model: str | None = None
    model_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    latency_seconds: float = 0.0
    budget: TokenBudget | None = None


class AutonomyRecommendation(BaseModel):
    """Deterministic routing recommendation (see ``autonomy.py``).

    A recommendation only — it is printed on the result and the claim
    packet, and nothing in the pipeline acts on it.
    """

    eligible_for_auto_approval: bool
    reasons: list[str] = Field(default_factory=list)


class TraceStep(BaseModel):
    """One pipeline step, with the actual output it produced.

    Additive field for the demo console: the UI renders these as the
    pipeline trace (extract → ingest → classify → retrieve → diagnose →
    options → draft → verify → validate → human approval) so a reviewer
    can inspect what each node really did, including its tool calls.
    """

    name: str  # extract | ingest | classify | retrieve | diagnose | options | draft | verify | review | validate | human_approval
    title: str
    status: str = "completed"  # completed | passed | failed | awaiting
    summary: str = ""
    details: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    shipment_id: str
    classification: Classification
    llm_suggestion: ClassificationSuggestion | None = None
    cross_check: ClassificationCrossCheck | None = None
    extractions: list[DocumentExtraction] = Field(default_factory=list)
    injection_flags: list[InjectionFlag] = Field(default_factory=list)
    diagnosis: Diagnosis | None = None
    recovery_options: list[RecoveryOption] = Field(default_factory=list)
    recommended_option_id: str | None = None
    delay_hours: float | None = None
    document_mismatches: list[DocumentMismatch] = Field(default_factory=list)
    document_check_warning: str | None = None
    policies: list[RetrievedPolicy] = Field(default_factory=list)
    draft: DraftOutput
    verification: VerificationResult | None = None
    review: ReviewerResult | None = None
    reviewer_blocked: bool = False
    validation: ValidationResult
    repair_attempted: bool = False
    repaired: bool = False
    repair_attempts: int = 0
    original_validation: ValidationResult | None = None
    autonomy: AutonomyRecommendation | None = None
    needs_information: bool = False
    information_request: InformationRequest | None = None
    telemetry: RunTelemetry | None = None
    trace: list[TraceStep] = Field(default_factory=list)
    approval_status: str = "awaiting_approval"
    decided_by: str | None = None  # who approved/rejected, once decided
    dispatch_status: str | None = None  # sent | failed — only when ACTION_WEBHOOK_URL is set
    external_action_taken: bool = False
    disclaimer: str = (
        "Reference prototype running on synthetic data. No external system "
        "was contacted and no customer message was sent. Draft requires "
        "human approval before any action."
    )
