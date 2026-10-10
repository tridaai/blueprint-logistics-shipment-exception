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
from .config import env_float, env_int, env_str, load_dotenv
from .graph import resume_approval, run_shipment
from .model_backends import ModelBackend, get_backend
from .object_store import get_object_store
from .ports import Checkpointer, EventSink, ObjectStore
from .retriever import Retriever, get_retriever
from .schemas import AgentResult, ShipmentInput
from .store import ApprovalRecord, ApprovalStore, carrier_summary, default_store

__all__ = ["ApprovalRecord", "BatchItem", "ShipmentService"]


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string (record timestamps)."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


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
    The schedule is bookkeeping, recorded on the ledger entry as
    ``next_retry_at``; who acts on it (an operator pressing retry,
    a scheduler polling :meth:`ShipmentService.due_dispatch_retries`)
    is the deployment's choice.
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

    def _resume_thread(self, shipment_id: str, decision: dict) -> None:
        """Complete the checkpointed graph thread with a decision.

        Best-effort by design: the store record the caller just saved
        is the decision of record. A thread that is missing or already
        finished (the analysis ran with checkpoints off, modes were
        mixed) has nothing to resume, and a resume failure never
        unmakes a recorded human decision.
        """
        checkpointer = self._get_checkpointer()
        if checkpointer is None:
            return
        try:
            resume_approval(shipment_id, decision, checkpointer)
        except Exception:
            pass

    def analyze(
        self, shipment: ShipmentInput | dict, event_sink: EventSink | None = None
    ) -> AgentResult:
        # Backend and retriever come from the environment (MODEL_BACKEND /
        # RETRIEVER, with the repo-root .env loaded) unless injected.
        # ``event_sink`` (optional) receives the run's structured events
        # (see events.py) — the API's streaming endpoint passes one.
        backend = self.backend or get_backend()
        if self.retriever is None:
            self.retriever = get_retriever()
        model = (
            shipment
            if isinstance(shipment, ShipmentInput)
            else ShipmentInput.model_validate(shipment)
        )
        return self._analyze_model(model, backend, self.retriever, event_sink=event_sink)

    def _analyze_model(
        self,
        model: ShipmentInput,
        backend: ModelBackend,
        retriever: Retriever,
        event_sink: EventSink | None = None,
    ) -> AgentResult:
        # Key-only documents get their text through the object-store
        # port before anything reads them (single + batch paths share
        # this entry point).
        model = self._resolve_document_texts(model)
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
            event_sink=event_sink,
            checkpointer=self._get_checkpointer(),
            thread_id=model.shipment_id,
        )
        self._get_store().save(
            ApprovalRecord(
                result=result,
                shipment=self._archive_documents(model),
                created_at=_now_iso(),
            )
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
        )
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
        record.decided_at = _now_iso()
        record.result.approval_status = "rejected"
        record.result.decided_by = reviewer
        record.result.decision_reason = reason or None
        record.result.external_action_taken = False
        store.save(record)
        self._resume_thread(
            shipment_id,
            {"decision": "rejected", "actor": reviewer, "reason": reason},
        )
        return record.result

    def dispatch_ledger(self, shipment_id: str) -> dict | None:
        """The webhook delivery ledger for one shipment, or None.

        The durable account of the approval's output routing: every
        attempt (when, outcome, HTTP status, signature id, error),
        the current status, and the retry bookkeeping (attempts used
        / remaining, when the next retry falls due). ``None`` for an
        unknown shipment; a known shipment with no webhook configured
        reports an empty ledger, not an error.
        """
        record = self._get_store().get(shipment_id)
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
        self, shipment_id: str, *, force: bool = False, now=None
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
        """
        from datetime import datetime, timezone

        store = self._get_store()
        record = store.get(shipment_id)
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

    def due_dispatch_retries(self, now=None) -> list[str]:
        """Shipment ids whose failed delivery is due a retry now.

        A record qualifies when its latest dispatch failed, attempts
        remain in the budget, and the recorded backoff has elapsed.
        This is the poll a scheduler/worker loop calls; performing
        the retries stays with :meth:`retry_dispatch`, one bounded
        attempt per call.
        """
        from datetime import datetime, timezone

        moment = now or datetime.now(timezone.utc)
        due: list[str] = []
        for record in self._get_store().records():
            if record.dispatch_status != "failed" or not record.dispatch_attempts:
                continue
            if len(record.dispatch_attempts) >= webhook_max_attempts():
                continue
            due_at = _parse_iso(record.dispatch_attempts[-1].get("next_retry_at"))
            if due_at is None or moment >= due_at:
                due.append(record.result.shipment_id)
        return due

    def get(self, shipment_id: str) -> AgentResult | None:
        record = self._get_store().get(shipment_id)
        return record.result if record else None
