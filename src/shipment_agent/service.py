"""Service layer: run + approval bookkeeping.

Approving a draft records the decision and marks the claim packet
ready. By default it performs NO external action (no email, no carrier
API call). The one opt-in exception is output routing: when
``ACTION_WEBHOOK_URL`` is configured, the approved packet is POSTed to
that endpoint — the customer's system of choice. Every delivery
attempt is written to the record's **delivery ledger** (timestamp,
HTTP status, signature id, error, next-retry due time), and a failed
delivery can be retried under a bounded attempt budget with
exponential backoff (see ``dispatch_approval_webhook``,
``retry_dispatch``, ``due_dispatch_retries``).

Records live in an approval store (``store.py``): PostgreSQL when
``DATABASE_URL`` is configured, so analyses and human decisions
survive restarts and are shared across replicas; the SQLite and
in-memory stores remain available as test doubles.

The approval gate is checkpointed (``checkpoints.py``, on unless
``CHECKPOINTS=off``): an analysis pauses its graph at the gate with
the state persisted under the shipment id, and approve/reject resume
that thread with the decision. The split is deliberate — the store is
the record of decisions (and the only thing the decision flow reads);
the checkpointer holds graph state. A missing/finished thread never
blocks a decision.

**Tenancy.** The store is partitioned by tenant (``store.py``):
every method here takes an optional ``tenant_id`` and resolves it
with :func:`resolve_tenant_id` — explicit argument, else the
``TENANT_ID`` environment default, else the default tenant — and
every store read is scoped to the resolved tenant, so one tenant's
shipments, memory, queue, and scorecards are invisible to another.
The gate's checkpoint threads are namespaced by tenant for the same
reason (``<tenant>:<shipment_id>``), and retrieval is scoped too:
each run resolves its retriever's per-tenant view (see
:func:`_scoped_retriever`), so a tenant's runs cite the shared
policy corpus plus its own SOPs, and no other tenant's. The one
deliberate exception is the dispatch-retry sweep: an operator
process that works every tenant's failed deliveries, resolving each
record's own tenant per retry.

**SLA breach events.** The queue flags a shipment whose wait has
blown its severity's age budget; :meth:`ShipmentService.sla_breach_sweep`
is the documented observer that acts on the flag — the first time a
sweep sees a breach, it fires one signed ``sla_breach`` webhook
event (opt-in via ``SLA_BREACH_WEBHOOK``), ledgered on the record's
own SLA ledger and deduped by its ``sla_breach_event_at`` marker.
Operators run the sweep on a timer (``shipment-agent sla-sweep``,
or ``POST /queue/sla-sweep`` for one tenant) next to the retry
worker; no request path ever blocks on it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .checkpoints import get_checkpointer
from .config import DEFAULT_TENANT_ID, env_float, env_int, env_str, load_dotenv
from .events import RunEvent
from .graph import resume_approval, run_shipment
from .insights import (
    all_carrier_scorecards,
    approval_queue as _approval_queue,
)
from .insights import (
    carrier_scorecard as _carrier_scorecard,
)
from .insights import (
    fleet_baseline as _fleet_baseline,
)
from .insights import (
    lane_baseline as _lane_baseline,
)
from .insights import (
    lane_scorecard as _lane_scorecard,
)
from .model_backends import ModelBackend, get_backend
from .object_store import get_object_store
from .ports import Checkpointer, EventSink, ObjectStore
from .retriever import Retriever, get_retriever
from .schemas import AgentResult, ShipmentInput
from .store import ApprovalRecord, ApprovalStore, carrier_summary, default_store

__all__ = [
    "ApprovalRecord",
    "BatchItem",
    "DecisionConflictError",
    "ShipmentService",
    "checkpoint_thread_id",
    "resolve_tenant_id",
]


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string (record timestamps)."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def resolve_tenant_id(tenant_id: str | None = None) -> str:
    """Resolve the tenant a call operates on.

    Precedence: the explicit argument (the API passes the
    ``X-Tenant-ID`` header), then the ``TENANT_ID`` environment
    default (a single-client deployment pins its tenant there), then
    :data:`~shipment_agent.config.DEFAULT_TENANT_ID`. Whitespace is
    never significant; an empty value is no value.
    """
    load_dotenv()
    for candidate in (tenant_id, env_str("TENANT_ID")):
        if candidate and candidate.strip():
            return candidate.strip()
    return DEFAULT_TENANT_ID


def checkpoint_thread_id(tenant_id: str, shipment_id: str) -> str:
    """The gate checkpoint thread for one tenant's shipment.

    Namespaced by tenant so two tenants' same-id shipments never
    share a graph thread (and one tenant's decision can never resume
    another's run)."""
    return f"{tenant_id}:{shipment_id}"


def _scoped_retriever(retriever: Retriever, tenant_id: str) -> Retriever:
    """The retriever view for one run's tenant.

    Retrieval was the last surface still shared across tenants:
    the corpus is tagged (shared documents plus each tenant's own
    SOPs), and a retriever that offers ``for_tenant`` (the shipped
    keyword / semantic / hybrid implementations do) is resolved to
    the run's tenant view here, so the run retrieves the shared
    corpus plus its own tenant's documents — another tenant's SOPs
    are absent from the corpus, not filtered from the results. A
    retriever without the capability (an injected double, a custom
    port implementation) is used exactly as provided: the seam's
    contract is the injector's to honour.
    """
    for_tenant = getattr(retriever, "for_tenant", None)
    if callable(for_tenant):
        return for_tenant(tenant_id)
    return retriever


@dataclass
class BatchItem:
    """One outcome of a batch analysis: a result, or the error that
    stopped that one shipment — never the whole batch."""

    shipment_id: str
    result: AgentResult | None = None
    error: str | None = None


def sign_webhook_body(body: bytes, secret: str) -> str:
    """The ``X-Trida-Signature`` value for a webhook body:
    ``sha256=<HMAC-SHA256 hex>`` keyed by ``ACTION_WEBHOOK_SECRET``.

    The receiver recomputes the same HMAC over the raw request body
    with the shared secret and compares (in constant time) — that
    proves the packet came from this agent and was not altered in
    transit, which matters because the receiver may act on it.
    """
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class DecisionConflictError(ValueError):
    """An Idempotency-Key was spent on the *opposite* decision.

    A subclass of ValueError so existing refusal handling keeps
    working, but distinct: the API maps it to 409 Conflict (the key
    names an operation that already happened, differently), where a
    plain re-decision attempt is a 422.
    """


@dataclass
class DispatchOutcome:
    """One webhook delivery attempt's result, in ledger detail.

    ``http_status`` is the response code when the endpoint answered
    at all (including an error status — a 500 is a *failed* delivery
    with a status, not a mystery); ``error`` names what went wrong
    (``"HTTP 500"``, or the transport exception). ``signature_id`` is
    the ``X-Trida-Signature`` value the delivery carried, when
    signing is configured — the identifier a receiver quotes when
    reporting a delivery, and the value it verifies against.
    """

    status: str  # sent | failed
    http_status: int | None = None
    signature_id: str | None = None
    error: str | None = None


def webhook_max_attempts() -> int:
    """Total dispatch attempts per approval (first try + retries).

    ``ACTION_WEBHOOK_MAX_ATTEMPTS``, default 3, minimum 1 — the bound
    that keeps a dead endpoint from being retried forever.
    """
    load_dotenv()
    return max(1, env_int("ACTION_WEBHOOK_MAX_ATTEMPTS", 3))


def webhook_retry_delay(attempt_number: int) -> float:
    """Seconds to wait after attempt ``attempt_number`` fails.

    Exponential backoff on ``ACTION_WEBHOOK_RETRY_BASE_SECONDS``
    (default 30): attempt 1 → base, attempt 2 → 2×base, and so on.
    The schedule is recorded on the ledger entry as
    ``next_retry_at``; the shipped worker loop
    (:meth:`ShipmentService.dispatch_retry_worker`, CLI
    ``dispatch-retries``) acts on it, and an operator can still
    retry by hand through the API.
    """
    load_dotenv()
    base = env_float("ACTION_WEBHOOK_RETRY_BASE_SECONDS", 30.0)
    return base * (2 ** max(0, attempt_number - 1))


def _parse_iso(value: str | None):
    """Parse an ISO-8601 timestamp this codebase wrote, or None."""
    if not value:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _deliver_webhook_payload(payload: dict, url: str) -> DispatchOutcome:
    """POST one JSON payload to a webhook URL, signed when configured.

    The transport shared by every outbound event (the approval
    packet, the SLA breach event): when ``ACTION_WEBHOOK_SECRET``
    is set the delivery is signed (``X-Trida-Signature``, see
    :func:`sign_webhook_body`) whichever URL the event targets.
    A 2xx is ``sent``; a non-2xx or any transport error is
    ``failed`` with the status/error recorded — delivery never
    raises, and the caller ledgers the outcome.
    """
    load_dotenv()
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    signature_id = None
    secret = env_str("ACTION_WEBHOOK_SECRET")
    if secret:
        signature_id = sign_webhook_body(body, secret)
        headers["X-Trida-Signature"] = signature_id
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    timeout = env_float("ACTION_WEBHOOK_TIMEOUT_SECONDS", 5.0)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            if 200 <= status < 300:
                return DispatchOutcome(
                    status="sent", http_status=status, signature_id=signature_id
                )
            return DispatchOutcome(
                status="failed",
                http_status=status,
                signature_id=signature_id,
                error=f"HTTP {status}",
            )
    except urllib.error.HTTPError as exc:  # the endpoint answered, with an error
        return DispatchOutcome(
            status="failed",
            http_status=int(exc.code),
            signature_id=signature_id,
            error=f"HTTP {exc.code}",
        )
    except Exception as exc:  # transport failure: recorded, never raised
        return DispatchOutcome(
            status="failed",
            signature_id=signature_id,
            error=f"{type(exc).__name__}: {exc}",
        )


def attempt_webhook_dispatch(record: ApprovalRecord) -> DispatchOutcome | None:
    """Make ONE approval-webhook delivery attempt.

    Returns ``None`` when no URL is configured (the default: no
    external action at all) — no attempt happened, so nothing is
    ledgered. Otherwise returns the outcome in full: a 2xx is
    ``sent``; a non-2xx or any transport error is ``failed`` with
    the status/error recorded — a failed dispatch never raises and
    never undoes the approval it follows.
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
    return _deliver_webhook_payload(payload, url)


# ---------------------------------------------------------------------------
# SLA breach events: the queue's own alarm, on the same signed channel
# ---------------------------------------------------------------------------


def sla_breach_webhook_enabled() -> bool:
    """Whether SLA breach events fire at all (opt-in, default off).

    ``SLA_BREACH_WEBHOOK=on`` turns the queue sweep's observations
    into outbound events. Off (the default), the sweep still reports
    the breaches it sees — it just sends nothing, like every other
    external action in this codebase: configured, never assumed.
    """
    load_dotenv()
    return (env_str("SLA_BREACH_WEBHOOK") or "").strip().lower() in (
        "on",
        "1",
        "true",
        "yes",
    )


def sla_breach_webhook_url() -> str | None:
    """Where SLA breach events are delivered.

    ``SLA_BREACH_WEBHOOK_URL`` when set (a paging/on-call endpoint
    separate from the system of record), else the approval
    webhook's ``ACTION_WEBHOOK_URL`` — one receiver may take both
    event shapes and route on the body's ``event`` field.
    """
    load_dotenv()
    return env_str("SLA_BREACH_WEBHOOK_URL") or env_str("ACTION_WEBHOOK_URL")


def build_sla_breach_payload(
    record: ApprovalRecord, queue_item: dict, detected_at: str
) -> dict:
    """The ``sla_breach`` event body for one breaching queue item.

    Everything a receiver needs to act without calling back: which
    shipment, whose tenant, what severity and budget, how long it
    has waited and by how much it is over, and the lane/carrier
    context from the queue item itself.
    """
    return {
        "event": "sla_breach",
        "shipment_id": record.result.shipment_id,
        "tenant_id": record.tenant_id,
        "exception_type": queue_item["exception_type"],
        "severity": queue_item["severity"],
        "carrier": queue_item["carrier"],
        "lane": queue_item["lane"],
        "sla_hours": queue_item["sla_hours"],
        "age_seconds": queue_item["age_seconds"],
        "overdue_seconds": queue_item["sla_overdue_seconds"],
        "created_at": record.created_at,
        "detected_at": detected_at,
    }


def attempt_sla_breach_dispatch(
    record: ApprovalRecord, queue_item: dict, detected_at: str
) -> DispatchOutcome | None:
    """Make ONE ``sla_breach`` delivery attempt for a breaching record.

    Returns ``None`` when no URL resolves (see
    :func:`sla_breach_webhook_url`) — no attempt happened, so
    nothing is ledgered and the breach is not marked as fired.
    Signed exactly like the approval packet when
    ``ACTION_WEBHOOK_SECRET`` is configured.
    """
    url = sla_breach_webhook_url()
    if not url:
        return None
    payload = build_sla_breach_payload(record, queue_item, detected_at)
    return _deliver_webhook_payload(payload, url)


def record_sla_attempt(
    record: ApprovalRecord, outcome: DispatchOutcome, *, at: str
) -> dict:
    """Ledger one ``sla_breach`` delivery attempt on the record.

    The event keeps its OWN ledger (``sla_dispatch_attempts``): the
    approval packet's ``dispatch_attempts`` / ``dispatch_status``
    describe that delivery alone and are never touched here. The
    first attempt also stamps ``sla_breach_event_at`` — the marker
    the sweep dedupes on, so the event fires once per shipment per
    analysis however often the sweep runs. Returns the entry.
    """
    entry = {
        "attempt": len(record.sla_dispatch_attempts) + 1,
        "at": at,
        "outcome": outcome.status,
        "http_status": outcome.http_status,
        "signature_id": outcome.signature_id,
        "error": outcome.error,
    }
    record.sla_dispatch_attempts.append(entry)
    if record.sla_breach_event_at is None:
        record.sla_breach_event_at = at
    return entry


def record_dispatch_attempt(record: ApprovalRecord, outcome: DispatchOutcome) -> dict:
    """Append one attempt to the record's delivery ledger.

    The ledger entry is the delivery's audit record: which attempt,
    when, the outcome, the HTTP status and signature id when there
    are any, the error when it failed, and — for a failure with
    attempts remaining — when the next retry falls due under the
    backoff schedule. The record's ``dispatch_status`` (and its echo
    on the result, plus ``external_action_taken``) follows the
    latest attempt. Returns the entry.
    """
    from datetime import timedelta, timezone, datetime

    attempt_number = len(record.dispatch_attempts) + 1
    now = datetime.now(timezone.utc)
    next_retry_at = None
    if outcome.status == "failed" and attempt_number < webhook_max_attempts():
        due = now + timedelta(seconds=webhook_retry_delay(attempt_number))
        next_retry_at = due.isoformat()
    entry = {
        "attempt": attempt_number,
        "at": now.isoformat(),
        "outcome": outcome.status,
        "http_status": outcome.http_status,
        "signature_id": outcome.signature_id,
        "error": outcome.error,
        "next_retry_at": next_retry_at,
    }
    record.dispatch_attempts.append(entry)
    record.dispatch_status = outcome.status
    record.result.dispatch_status = outcome.status
    record.result.external_action_taken = outcome.status == "sent"
    return entry


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

    When ``ACTION_WEBHOOK_SECRET`` is set, the delivery is signed
    (``X-Trida-Signature``, see :func:`sign_webhook_body`); unset,
    deliveries are unsigned, exactly as before.

    The attempt is also written to the record's delivery ledger
    (:func:`record_dispatch_attempt`) — that ledger, not this return
    value, is the durable account of the delivery.
    """
    outcome = attempt_webhook_dispatch(record)
    if outcome is None:
        return None
    record_dispatch_attempt(record, outcome)
    return outcome.status


@dataclass
class ShipmentService:
    backend: ModelBackend | None = None
    retriever: Retriever | None = None
    store: ApprovalStore | None = None
    # The gate checkpointer: a saver instance, None (= resolve from the
    # environment, CHECKPOINTS, default on), or False (= disabled,
    # whatever the environment says — the CLI batch path uses this
    # because it persists nothing).
    checkpointer: Checkpointer | bool | None = None
    # The document object store: an instance, None (= resolve from
    # the environment, S3_BUCKET), or False (= disabled, whatever the
    # environment says — tests and offline paths use this).
    object_store: ObjectStore | bool | None = None
    _resolved_store: ApprovalStore | None = field(default=None, repr=False)
    _resolved_checkpointer: Checkpointer | None = field(default=None, repr=False)
    _checkpointer_resolved: bool = field(default=False, repr=False)
    _resolved_object_store: ObjectStore | None = field(default=None, repr=False)
    _object_store_resolved: bool = field(default=False, repr=False)

    def _get_store(self) -> ApprovalStore:
        if self._resolved_store is None:
            self._resolved_store = self.store or default_store()
        return self._resolved_store

    def _get_checkpointer(self) -> Checkpointer | None:
        if not self._checkpointer_resolved:
            self._checkpointer_resolved = True
            if self.checkpointer is False:
                self._resolved_checkpointer = None
            elif self.checkpointer is not None:
                self._resolved_checkpointer = self.checkpointer  # type: ignore[assignment]
            else:
                self._resolved_checkpointer = get_checkpointer()
        return self._resolved_checkpointer

    def _get_object_store(self) -> ObjectStore | None:
        if not self._object_store_resolved:
            self._object_store_resolved = True
            if self.object_store is False:
                self._resolved_object_store = None
            elif self.object_store is not None:
                self._resolved_object_store = self.object_store  # type: ignore[assignment]
            else:
                self._resolved_object_store = get_object_store()
        return self._resolved_object_store

    def _resolve_document_texts(self, model: ShipmentInput) -> ShipmentInput:
        """Fetch document text for key-only documents (production intake).

        An adapter that already stored the document submits just its
        ``object_key``; the pipeline needs the text, so it is fetched
        through the port here — the one place raw bytes enter.
        A failed fetch leaves the document text empty (the run
        proceeds; the document checks see the gap), never raises.
        """
        store = self._get_object_store()
        if store is None:
            return model
        documents = []
        changed = False
        for doc in model.documents:
            if doc.object_key and not doc.raw_text:
                try:
                    text = store.get(doc.object_key).decode(
                        "utf-8", errors="replace"
                    )
                except Exception:
                    text = ""
                if text:
                    doc = doc.model_copy(update={"raw_text": text})
                    changed = True
            documents.append(doc)
        if not changed:
            return model
        return model.model_copy(update={"documents": documents})

    def _archive_documents(self, model: ShipmentInput) -> dict:
        """Archive inline documents to the object store.

        Returns the shipment payload to persist: the model's dump,
        with ``object_key`` filled in for every document that was
        written. Archiving is bookkeeping, not analysis — a failed
        put skips that document's key and never fails the run.
        """
        payload = model.model_dump(mode="json")
        store = self._get_object_store()
        if store is None:
            return payload
        for index, doc in enumerate(model.documents):
            if not doc.raw_text or doc.object_key:
                continue
            name = doc.document_id or f"doc-{index}"
            key = f"shipments/{model.shipment_id}/documents/{name}.txt"
            try:
                store.put(key, doc.raw_text.encode("utf-8"), "text/plain; charset=utf-8")
            except Exception:
                continue
            payload["documents"][index]["object_key"] = key
        return payload

    def _resume_thread(
        self, shipment_id: str, decision: dict, tenant_id: str = DEFAULT_TENANT_ID
    ) -> None:
        """Complete the checkpointed graph thread with a decision.

        Best-effort by design: the store record the caller just saved
        is the decision of record. A thread that is missing or already
        finished (the analysis ran with checkpoints off, modes were
        mixed) has nothing to resume, and a resume failure never
        unmakes a recorded human decision. The thread id is the
        tenant-namespaced one the analysis ran under.
        """
        checkpointer = self._get_checkpointer()
        if checkpointer is None:
            return
        try:
            resume_approval(checkpoint_thread_id(tenant_id, shipment_id), decision, checkpointer)
        except Exception:
            pass

    @staticmethod
    def _normalize_idempotency_key(key: str | None) -> str | None:
        """An idempotency key is an opaque caller string; whitespace
        is never significant, and an empty key is no key."""
        if key is None:
            return None
        key = key.strip()
        return key or None

    def idempotent_result(
        self,
        idempotency_key: str | None,
        shipment_id: str,
        tenant_id: str | None = None,
    ) -> AgentResult | None:
        """The stored run for (tenant, key, shipment id), flagged as
        a replay.

        The dedupe lookup behind :meth:`analyze`'s idempotency: a
        hit means this exact submission was already analysed, so the
        stored result — with the original run's telemetry — is the
        answer, and no model is called again. Returns None when there
        is no key or no stored run under it. The lookup is scoped to
        the resolved tenant: another tenant's run under the same key
        and shipment id is a different submission, not a replay.
        """
        key = self._normalize_idempotency_key(idempotency_key)
        if key is None:
            return None
        record = self._get_store().get_by_idempotency(
            key, shipment_id, tenant_id=resolve_tenant_id(tenant_id)
        )
        if record is None:
            return None
        return record.result.model_copy(update={"idempotent_replay": True})

    def analyze(
        self,
        shipment: ShipmentInput | dict,
        event_sink: EventSink | None = None,
        idempotency_key: str | None = None,
        tenant_id: str | None = None,
    ) -> AgentResult:
        # Backend and retriever come from the environment (MODEL_BACKEND /
        # RETRIEVER, with the repo-root .env loaded) unless injected.
        # ``event_sink`` (optional) receives the run's structured events
        # (see events.py) — the API's streaming endpoint passes one.
        #
        # ``idempotency_key`` (optional) makes the call safe to retry:
        # when a run is already stored under (key, shipment id), the
        # stored result is returned flagged ``idempotent_replay`` and
        # the pipeline does not run again — no duplicate model spend,
        # no second record, no reset of a decision already taken.
        #
        # ``tenant_id`` (optional) names the tenant partition the run
        # belongs to (see resolve_tenant_id for the precedence); the
        # record, its memory reads, and its idempotency scope all live
        # in that partition.
        model = (
            shipment
            if isinstance(shipment, ShipmentInput)
            else ShipmentInput.model_validate(shipment)
        )
        tenant = resolve_tenant_id(tenant_id)
        key = self._normalize_idempotency_key(idempotency_key)
        if key is not None:
            replay = self.idempotent_result(key, model.shipment_id, tenant_id=tenant)
            if replay is not None:
                return replay
        backend = self.backend or get_backend()
        if self.retriever is None:
            self.retriever = get_retriever()
        return self._analyze_model(
            model,
            backend,
            self.retriever,
            event_sink=event_sink,
            idempotency_key=key,
            tenant_id=tenant,
        )

    def _analyze_model(
        self,
        model: ShipmentInput,
        backend: ModelBackend,
        retriever: Retriever,
        event_sink: EventSink | None = None,
        idempotency_key: str | None = None,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> AgentResult:
        # Key-only documents get their text through the object-store
        # port before anything reads them (single + batch paths share
        # this entry point).
        model = self._resolve_document_texts(model)
        # Memory: what the store already knows about this consignee and
        # this lane becomes diagnosis evidence for the new analysis.
        # The raw entries ride along too — the diagnosis tool loop
        # (provider mode) reads lane / carrier history through them.
        # Every read is scoped to this run's tenant: memory is a
        # tenant's own history, never another tenant's.
        store = self._get_store()
        priors = store.prior_shipments(
            exclude_shipment_id=model.shipment_id, tenant_id=tenant_id
        )
        history = self._history_summary(model, priors)
        # Feedback loop: what human deciders said about earlier cases
        # for this consignee / lane (with their reasons) joins the same
        # evidence — the next diagnosis shows the pattern of oversight.
        feedback = self._feedback_for(model, tenant_id=tenant_id)
        if feedback:
            history = {**(history or {}), "feedback": feedback}
        # Carrier scorecard: the carrier's whole stored track record
        # (exception mix, damage rate, human approval rate) joins the
        # evidence too, even when the consignee/lane memory is empty —
        # a carrier's history is a decision input on its own. The
        # fleet baseline rides along: it is what the card's rates are
        # compared against when option scoring applies its carrier
        # reliability term (options.reliability_adjustment). Both are
        # computed over the priors only (this shipment excluded).
        records = store.records(tenant_id=tenant_id)
        card = _carrier_scorecard(
            records,
            model.carrier,
            exclude_shipment_id=model.shipment_id,
        )
        if card is not None:
            history = {**(history or {}), "carrier_scorecard": card}
            baseline = _fleet_baseline(
                [r for r in records if r.result.shipment_id != model.shipment_id]
            )
            if baseline is not None:
                history = {**history, "fleet_baseline": baseline}
            # The lane axis of the same record: this carrier on THIS
            # corridor, against the corridor's own baseline. When the
            # lane history is thick enough, those are the figures the
            # reliability term reads (options._reliability_basis) —
            # a carrier fine everywhere except one lane should be
            # priced on the lane the freight is about to travel.
            lane_card = _lane_scorecard(
                records,
                model.carrier,
                model.origin,
                model.destination,
                exclude_shipment_id=model.shipment_id,
            )
            if lane_card is not None:
                history = {**history, "carrier_lane_scorecard": lane_card}
                lane_base = _lane_baseline(
                    records,
                    model.origin,
                    model.destination,
                    exclude_shipment_id=model.shipment_id,
                )
                if lane_base is not None:
                    history = {**history, "lane_baseline": lane_base}
        result = run_shipment(
            model,
            backend=backend,
            retriever=_scoped_retriever(retriever, tenant_id),
            history=history,
            priors=priors,
            event_sink=event_sink,
            checkpointer=self._get_checkpointer(),
            thread_id=checkpoint_thread_id(tenant_id, model.shipment_id),
        )
        store.save(
            ApprovalRecord(
                result=result,
                tenant_id=tenant_id,
                shipment=self._archive_documents(model),
                created_at=_now_iso(),
                idempotency_key=idempotency_key,
            )
        )
        return result

    def analyze_batch(
        self,
        shipments: list[ShipmentInput | dict],
        concurrency: int = 1,
        tenant_id: str | None = None,
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
        it never fails the batch. The whole batch lands in one tenant
        partition (``tenant_id``, resolved like everywhere else).
        """
        tenant = resolve_tenant_id(tenant_id)

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
                result = self._analyze_model(model, backend, retriever, tenant_id=tenant)
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

    def _feedback_for(
        self, shipment: ShipmentInput, tenant_id: str = DEFAULT_TENANT_ID
    ) -> list[dict]:
        """Recent reviewer feedback matching this consignee or lane.

        Bounded to the last 3 matching decisions (the store returns
        newest first), reasons truncated — feedback is evidence, not a
        transcript. Each entry: {decision, reason, match} where match
        is "consignee" or "lane" (consignee wins when both match).
        Reads only the run's own tenant partition.
        """
        entries = self._get_store().decision_feedback(
            exclude_shipment_id=shipment.shipment_id, tenant_id=tenant_id
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
        self,
        shipment: ShipmentInput,
        priors: list[dict] | None = None,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> dict | None:
        """Summarise prior analysed shipments for this consignee + lane.

        Counts only priors that themselves had an exception, with the
        most recent exception types — the diagnosis cites them as
        evidence ("2 prior exceptions for this consignee in the stored
        history"). Returns None when there is nothing to remember.
        """
        if priors is None:
            priors = self._get_store().prior_shipments(
                exclude_shipment_id=shipment.shipment_id, tenant_id=tenant_id
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

    @staticmethod
    def _decision_replay(
        record: ApprovalRecord, key: str | None, requested: str
    ) -> AgentResult:
        """Answer a decision call for an already-decided record.

        Three cases, mirroring the analyze idempotency contract one
        level up:

        - the record's decision was made under this same
          ``Idempotency-Key`` and the caller asks for the *same*
          decision again → the recorded decision, flagged
          ``idempotent_replay``. Nothing re-fires: no second webhook
          dispatch, no second thread resume, and the feedback loop
          (derived from the record) sees the decision exactly once.
          First write wins — a replay carrying a different actor or
          reason still returns the decision as recorded.
        - same key, *opposite* decision → :class:`DecisionConflictError`:
          the key names an operation that already happened,
          differently (the API maps this to 409).
        - anything else (no key, a different key, an unkeyed
          decision) → the long-standing refusal: the shipment is
          not awaiting approval.
        """
        decided_as = record.result.approval_status
        if key is not None and record.decision_idempotency_key == key:
            if decided_as == requested:
                return record.result.model_copy(update={"idempotent_replay": True})
            raise DecisionConflictError(
                f"Idempotency-Key {key!r} already decided shipment "
                f"{record.result.shipment_id} as {decided_as!r}; it cannot "
                f"also decide it as {requested!r}"
            )
        raise ValueError(
            f"Shipment {record.result.shipment_id} is not awaiting approval "
            f"(status: {decided_as})"
        )

    def approve(
        self,
        shipment_id: str,
        approver: str,
        reason: str = "",
        tenant_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> AgentResult:
        store = self._get_store()
        record = store.get(shipment_id, tenant_id=resolve_tenant_id(tenant_id))
        if record is None:
            raise KeyError(f"Unknown shipment_id: {shipment_id} (analyze it first)")
        key = self._normalize_idempotency_key(idempotency_key)
        if record.result.approval_status != "awaiting_approval":
            return self._decision_replay(record, key, "approved")
        if not record.result.validation.passed:
            raise ValueError(
                "Draft failed guardrail validation and cannot be approved: "
                + "; ".join(record.result.validation.errors)
            )
        record.decision_idempotency_key = key
        record.approved = True
        record.approver = approver
        record.approve_reason = reason
        record.decided_at = _now_iso()
        record.result.approval_status = "approved"
        record.result.decided_by = approver
        record.result.decision_reason = reason or None
        # Output routing: by default there is NO external action. When
        # the operator configures ACTION_WEBHOOK_URL, the approved
        # packet is POSTed to that endpoint (the customer's system of
        # choice) and the attempt is written to the record's delivery
        # ledger — a failed dispatch leaves the approval standing,
        # says so on the result, and can be retried (retry_dispatch).
        record.result.external_action_taken = False
        dispatch_approval_webhook(record)
        store.save(record)
        self._resume_thread(
            shipment_id,
            {"decision": "approved", "actor": approver, "reason": reason},
            tenant_id=record.tenant_id,
        )
        return record.result

    def reject(
        self,
        shipment_id: str,
        reviewer: str,
        reason: str = "",
        tenant_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> AgentResult:
        store = self._get_store()
        record = store.get(shipment_id, tenant_id=resolve_tenant_id(tenant_id))
        if record is None:
            raise KeyError(f"Unknown shipment_id: {shipment_id} (analyze it first)")
        key = self._normalize_idempotency_key(idempotency_key)
        if record.result.approval_status != "awaiting_approval":
            return self._decision_replay(record, key, "rejected")
        record.decision_idempotency_key = key
        record.rejected_by = reviewer
        record.reject_reason = reason
        record.decided_at = _now_iso()
        record.result.approval_status = "rejected"
        record.result.decided_by = reviewer
        record.result.decision_reason = reason or None
        record.result.external_action_taken = False
        store.save(record)
        self._resume_thread(
            shipment_id,
            {"decision": "rejected", "actor": reviewer, "reason": reason},
            tenant_id=record.tenant_id,
        )
        return record.result

    def dispatch_ledger(
        self, shipment_id: str, tenant_id: str | None = None
    ) -> dict | None:
        """The webhook delivery ledger for one shipment, or None.

        The durable account of the approval's output routing: every
        attempt (when, outcome, HTTP status, signature id, error),
        the current status, and the retry bookkeeping (attempts used
        / remaining, when the next retry falls due). ``None`` for an
        unknown shipment — including one that exists only in another
        tenant's partition; a known shipment with no webhook
        configured reports an empty ledger, not an error.
        """
        record = self._get_store().get(
            shipment_id, tenant_id=resolve_tenant_id(tenant_id)
        )
        if record is None:
            return None
        attempts = record.dispatch_attempts
        latest = attempts[-1] if attempts else None
        return {
            "shipment_id": shipment_id,
            "dispatch_status": record.dispatch_status,
            "attempts": attempts,
            "attempts_used": len(attempts),
            "attempts_max": webhook_max_attempts(),
            "attempts_remaining": max(0, webhook_max_attempts() - len(attempts)),
            "next_retry_at": latest.get("next_retry_at") if latest else None,
        }

    def retry_dispatch(
        self,
        shipment_id: str,
        *,
        force: bool = False,
        now=None,
        tenant_id: str | None = None,
    ) -> AgentResult:
        """Retry a failed approval-webhook delivery, once.

        The retry is bounded and honest about its bounds: the record
        must exist and be approved, a webhook must be configured, the
        delivery must not already be ``sent``, the attempt budget
        (``ACTION_WEBHOOK_MAX_ATTEMPTS`` total attempts) must have
        room, and the backoff recorded on the last failed attempt
        must have elapsed — unless ``force`` is set, the operator
        override for "I fixed the endpoint, send it now". The new
        attempt joins the ledger either way it lands; a retry never
        re-records the decision and never resumes the graph thread —
        the approval happened once, this is only its delivery.

        ``tenant_id`` scopes the record lookup like every other read;
        the retry sweep passes each due record's own tenant (see
        :meth:`run_dispatch_retries_once`).
        """
        from datetime import datetime, timezone

        store = self._get_store()
        record = store.get(shipment_id, tenant_id=resolve_tenant_id(tenant_id))
        if record is None:
            raise KeyError(f"Unknown shipment_id: {shipment_id} (analyze it first)")
        if record.result.approval_status != "approved":
            raise ValueError(
                f"Shipment {shipment_id} is not approved (status: "
                f"{record.result.approval_status}) — only an approval dispatches"
            )
        load_dotenv()
        if not env_str("ACTION_WEBHOOK_URL"):
            raise ValueError(
                "ACTION_WEBHOOK_URL is not configured — there is no delivery to retry"
            )
        if record.dispatch_status == "sent":
            raise ValueError(
                f"Shipment {shipment_id}'s packet was already delivered (dispatch_status: sent)"
            )
        attempts_used = len(record.dispatch_attempts)
        if attempts_used >= webhook_max_attempts():
            raise ValueError(
                f"Dispatch retry budget exhausted for {shipment_id}: "
                f"{attempts_used} attempt(s) already made "
                f"(ACTION_WEBHOOK_MAX_ATTEMPTS={webhook_max_attempts()})"
            )
        if attempts_used and not force:
            due_at = _parse_iso(record.dispatch_attempts[-1].get("next_retry_at"))
            moment = now or datetime.now(timezone.utc)
            if due_at is not None and moment < due_at:
                raise ValueError(
                    f"Backoff has not elapsed for {shipment_id}: next retry is due "
                    f"at {record.dispatch_attempts[-1]['next_retry_at']} "
                    "(pass force to retry now)"
                )
        outcome = attempt_webhook_dispatch(record)
        if outcome is None:  # URL vanished between the check and the call
            raise ValueError(
                "ACTION_WEBHOOK_URL is not configured — there is no delivery to retry"
            )
        record_dispatch_attempt(record, outcome)
        store.save(record)
        return record.result

    def _due_retry_records(self, now=None) -> list[ApprovalRecord]:
        """Records (any tenant) whose failed delivery is due a retry.

        A record qualifies when its latest dispatch failed, attempts
        remain in the budget, and the recorded backoff has elapsed.
        The scan is deliberately unscoped — the retry worker is an
        operator process serving every tenant — and each record
        carries its own tenant for the scoped retry that follows.
        """
        from datetime import datetime, timezone

        moment = now or datetime.now(timezone.utc)
        due: list[ApprovalRecord] = []
        for record in self._get_store().records():
            if record.dispatch_status != "failed" or not record.dispatch_attempts:
                continue
            if len(record.dispatch_attempts) >= webhook_max_attempts():
                continue
            due_at = _parse_iso(record.dispatch_attempts[-1].get("next_retry_at"))
            if due_at is None or moment >= due_at:
                due.append(record)
        return due

    def due_dispatch_retries(self, now=None) -> list[str]:
        """Shipment ids whose failed delivery is due a retry now.

        This is the poll a scheduler/worker loop calls; performing
        the retries stays with :meth:`retry_dispatch`, one bounded
        attempt per call. (Ids alone cannot name a record across
        tenants — the sweep itself works from the records, see
        :meth:`run_dispatch_retries_once`.)
        """
        return [
            record.result.shipment_id for record in self._due_retry_records(now=now)
        ]

    def run_dispatch_retries_once(self, now=None) -> list[dict]:
        """Perform every due dispatch retry, one bounded attempt each.

        Returns one entry per due shipment: ``{"shipment_id",
        "outcome"}`` with the new dispatch status (``sent`` /
        ``failed``), or ``{"shipment_id", "error"}`` when a retry was
        refused between the due-scan and the attempt (the budget ran
        out, the record changed state) — a refusal is reported in the
        sweep's results, never raised: one bad record must not stop
        a sweep from reaching the rest of the queue.
        """
        from datetime import datetime, timezone

        moment = now or datetime.now(timezone.utc)
        outcomes: list[dict] = []
        for record in self._due_retry_records(now=moment):
            shipment_id = record.result.shipment_id
            try:
                # Each retry is scoped to the record's own tenant:
                # the sweep serves every tenant, but a retry still
                # reads and writes inside one partition.
                result = self.retry_dispatch(
                    shipment_id, now=moment, tenant_id=record.tenant_id
                )
            except (KeyError, ValueError) as exc:
                outcomes.append({"shipment_id": shipment_id, "error": str(exc)})
                continue
            outcomes.append(
                {"shipment_id": shipment_id, "outcome": result.dispatch_status}
            )
        return outcomes

    def dispatch_retry_worker(
        self,
        stop_event: threading.Event,
        *,
        interval_seconds: float = 60.0,
        now_fn=None,
        max_sweeps: int | None = None,
    ) -> dict:
        """The scheduler contract made real: loop the retry sweep.

        Sweeps :meth:`run_dispatch_retries_once`, waits
        ``interval_seconds``, and repeats until ``stop_event`` is set
        (or ``max_sweeps`` sweeps have run — the bound one-shot and
        cron-style invocations use). The wait is
        ``stop_event.wait``, so a stop request lands immediately
        rather than at the end of an interval; ``now_fn`` injects the
        clock the due-checks read (tests drive a fake one), defaulting
        to real UTC now. Returns the run's totals:
        ``{"sweeps", "attempted", "sent", "failed"}`` (a refused
        retry counts as attempted, under neither sent nor failed).

        Run it as its own process (``shipment-agent dispatch-retries``)
        next to the API — a sidecar, a systemd unit, or a cron-invoked
        ``--once`` — never inside the request path: an approval must
        not wait on a downstream endpoint's backoff schedule.
        """
        from datetime import datetime, timezone

        clock = now_fn or (lambda: datetime.now(timezone.utc))
        totals = {"sweeps": 0, "attempted": 0, "sent": 0, "failed": 0}
        while not stop_event.is_set():
            if max_sweeps is not None and totals["sweeps"] >= max_sweeps:
                break
            for entry in self.run_dispatch_retries_once(now=clock()):
                totals["attempted"] += 1
                if entry.get("outcome") == "sent":
                    totals["sent"] += 1
                elif entry.get("outcome") == "failed":
                    totals["failed"] += 1
            totals["sweeps"] += 1
            if max_sweeps is not None and totals["sweeps"] >= max_sweeps:
                break
            stop_event.wait(interval_seconds)
        return totals

    def approval_queue(self, tenant_id: str | None = None) -> list[dict]:
        """The approval queue: awaiting shipments, severity first,
        then oldest, each with the flags an approver scans for, its
        age bucket, and its SLA view under the configured budgets
        (``QUEUE_SLA_HOURS_<SEVERITY>``, see ``insights``). Scoped to
        the resolved tenant's partition — a tenant's approvers work
        their own queue."""
        from .insights import sla_thresholds_from_env

        return _approval_queue(
            self._get_store().records(tenant_id=resolve_tenant_id(tenant_id)),
            sla_hours=sla_thresholds_from_env(),
        )

    def approval_queue_summary(self, tenant_id: str | None = None) -> dict:
        """The queue's health summary (depth, SLA breaches, age and
        severity mix) — see ``insights.queue_summary``."""
        from .insights import queue_summary

        return queue_summary(self.approval_queue(tenant_id=tenant_id))

    def sla_breach_sweep(
        self,
        *,
        now=None,
        event_sink: EventSink | None = None,
        tenant_id: str | None = None,
    ) -> list[dict]:
        """Observe the queue; fire one ``sla_breach`` event per new breach.

        The sweep is the SLA story's actor, the way the retry worker
        is the delivery ledger's: the queue has always *flagged*
        breaches, and this is what makes one page somebody. For
        every awaiting shipment whose wait has blown its severity's
        budget and whose record has not yet fired, one signed
        ``sla_breach`` webhook event is delivered (see
        :func:`attempt_sla_breach_dispatch`) and ledgered on the
        record's own SLA ledger; the record's
        ``sla_breach_event_at`` marker dedupes, so repeated sweeps
        never re-fire — the event is "first observed", once per
        shipment per analysis.

        The feature is opt-in (``SLA_BREACH_WEBHOOK=on``): with it
        off, or with no webhook URL configured, the sweep still
        *reports* the breaches it observed (``outcome``:
        ``disabled`` / ``not_configured``) and marks nothing, so
        enabling the channel later fires for breaches still open.
        A failed delivery IS marked: the event fired once and its
        ledger entry carries the failure — the sweep is an alarm,
        not a retry loop (deliveries that must land use the approval
        packet's budget machinery instead).

        Scope: with ``tenant_id`` the sweep covers that tenant's
        queue (the API passes the caller's tenant); without one it
        covers every tenant's queue — the operator shape the CLI
        runs. A ``sla_breach`` stream event joins each firing when
        an ``event_sink`` is given. Returns one entry per newly
        observed breach: ``{"shipment_id", "tenant_id", "outcome",
        ...}``.
        """
        from datetime import datetime, timezone

        from .insights import approval_queue as _queue_projection
        from .insights import sla_thresholds_from_env

        moment = now or datetime.now(timezone.utc)
        detected_at = moment.isoformat()
        store = self._get_store()
        if tenant_id is not None:
            records = store.records(tenant_id=resolve_tenant_id(tenant_id))
        else:
            records = store.records()
        by_key = {
            (record.tenant_id, record.result.shipment_id): record
            for record in records
        }
        # The queue projection runs per tenant partition: severity /
        # age ordering is a within-tenant notion, and the sweep must
        # never compute one tenant's queue from another's records.
        partitions: dict[str, list[ApprovalRecord]] = {}
        for record in records:
            partitions.setdefault(record.tenant_id, []).append(record)
        thresholds = sla_thresholds_from_env()
        enabled = sla_breach_webhook_enabled()
        entries: list[dict] = []
        for partition_tenant, partition_records in partitions.items():
            items = _queue_projection(
                partition_records, now=moment, sla_hours=thresholds
            )
            for item in items:
                if not item["sla_breach"]:
                    continue
                record = by_key.get((partition_tenant, item["shipment_id"]))
                if record is None or record.sla_breach_event_at is not None:
                    continue  # already fired for this analysis — dedupe
                if record.result.approval_status != "awaiting_approval":
                    continue  # decided between projection and firing
                base = {
                    "shipment_id": item["shipment_id"],
                    "tenant_id": partition_tenant,
                    "severity": item["severity"],
                    "sla_hours": item["sla_hours"],
                    "overdue_seconds": item["sla_overdue_seconds"],
                }
                if not enabled:
                    entries.append({**base, "outcome": "disabled"})
                    continue
                outcome = attempt_sla_breach_dispatch(record, item, detected_at)
                if outcome is None:
                    entries.append({**base, "outcome": "not_configured"})
                    continue
                entry = record_sla_attempt(record, outcome, at=detected_at)
                store.save(record)
                if event_sink is not None:
                    event_sink.emit(
                        RunEvent(
                            type="sla_breach",
                            shipment_id=item["shipment_id"],
                            detail={
                                "tenant_id": partition_tenant,
                                "severity": item["severity"],
                                "sla_hours": item["sla_hours"],
                                "overdue_seconds": item["sla_overdue_seconds"],
                                "outcome": outcome.status,
                            },
                        )
                    )
                entries.append(
                    {
                        **base,
                        "outcome": outcome.status,
                        "http_status": outcome.http_status,
                        "signature_id": outcome.signature_id,
                        "error": outcome.error,
                        "attempt": entry["attempt"],
                    }
                )
        return entries

    def carrier_scorecards(self, tenant_id: str | None = None) -> list[dict]:
        """Scorecards for every carrier in the tenant's partition,
        busiest first."""
        return all_carrier_scorecards(
            self._get_store().records(tenant_id=resolve_tenant_id(tenant_id))
        )

    def carrier_scorecard(
        self, carrier: str, tenant_id: str | None = None
    ) -> dict | None:
        """One carrier's scorecard within the tenant's partition, or
        None when it has no history there."""
        return _carrier_scorecard(
            self._get_store().records(tenant_id=resolve_tenant_id(tenant_id)),
            carrier,
        )

    def get(
        self, shipment_id: str, tenant_id: str | None = None
    ) -> AgentResult | None:
        record = self._get_store().get(
            shipment_id, tenant_id=resolve_tenant_id(tenant_id)
        )
        return record.result if record else None
