"""Approval persistence: where analysis results and human decisions live.

PostgreSQL is the system of record: with ``DATABASE_URL`` set, records
live in the ``approvals`` table (migration ``0002``) via
:class:`PostgresStore`, so a human decision — and the diagnosis memory
built from earlier shipments — survives restarts and is shared across
service replicas. The SQLite and in-memory implementations remain as
**test doubles only**: the hermetic test suite injects them explicitly
(``STATE_DB_PATH`` still selects the SQLite double, for the persistence
tests and throwaway local files), and a run with neither variable set
falls back to the in-memory double — nothing persists, and the docs
say so wherever that mode is mentioned.

All stores also serve the reviewer feedback loop
(``decision_feedback``): decided records with non-empty reasons,
newest first, for the diagnosis of the next matching case.

Re-analyzing a shipment replaces its record and resets the decision,
matching the service's long-standing behaviour.

**Tenancy.** Every record carries a ``tenant_id`` and the store's
identity key is the pair ``(tenant_id, shipment_id)``: two tenants
may both have a ``SYN-1001``, and neither can see the other's. Every
read takes an optional ``tenant_id`` — a scoped read returns only
that tenant's partition (a cross-tenant ``get`` finds nothing, which
the API surfaces as a 404), and ``None`` means the operator's
unscoped view across all tenants, which only process-internal sweeps
(the dispatch retry worker) use. The service resolves the caller's
tenant (header / ``TENANT_ID`` / the default tenant) and always
reads scoped; see ``service.resolve_tenant_id``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_TENANT_ID, env_str, load_dotenv
from .schemas import AgentResult


@dataclass
class ApprovalRecord:
    result: AgentResult
    # The tenant this record belongs to. Stamped by the service at
    # analysis time from the caller's resolved tenant; rows written
    # before tenancy (and hand-built test records) sit in the default
    # tenant, which is where single-tenant deployments live anyway.
    tenant_id: str = DEFAULT_TENANT_ID
    shipment: dict | None = None  # the analysed ShipmentInput, for history lookups
    approver: str | None = None
    approved: bool = False
    approve_reason: str = ""
    rejected_by: str | None = None
    reject_reason: str = ""
    dispatch_status: str | None = None  # outbound webhook outcome, when configured
    # The webhook delivery ledger: one entry per dispatch attempt
    # (see service.record_dispatch_attempt) — {attempt, at, outcome,
    # http_status, signature_id, error, next_retry_at}. Empty when no
    # webhook is configured or no approval has dispatched yet.
    dispatch_attempts: list[dict] = field(default_factory=list)
    # The Idempotency-Key the analysis was submitted with, when the
    # caller sent one (see service.analyze). A repeat analyze with
    # the same key for the same shipment returns this record instead
    # of re-running the pipeline.
    idempotency_key: str | None = None
    # The id of the API key that authenticated the analysis request
    # (``<tenant>:current`` / ``<tenant>:previous`` / ``shared``) —
    # an identifier naming the key's generation, never the secret
    # itself (see service.analyze). The audit export carries it, so
    # a key rotation in progress is visible per record: analyses
    # still arriving under ``:previous`` are the old key's remaining
    # users. None when the API ran open or the caller was not the API.
    auth_key_id: str | None = None
    # SLA breach event bookkeeping (see service.sla_breach_sweep):
    # when the sweep first fired the record's signed ``sla_breach``
    # webhook event (None = never fired — the dedupe marker, so a
    # breach pages once, not on every sweep), and that event's own
    # delivery ledger — kept separate from ``dispatch_attempts``,
    # which belongs to the approval packet's delivery alone.
    sla_breach_event_at: str | None = None
    sla_dispatch_attempts: list[dict] = field(default_factory=list)
    # The wait (seconds) at the moment the breach event fired. The
    # escalation ladder's second rung exists for breaches that were
    # reported while still in rung-1 territory and then kept aging;
    # the sweep compares this stamp against the escalation threshold
    # to tell those apart from a breach first observed already past
    # it (whose receiver was told the full wait in the first event).
    sla_breach_age_seconds: float | None = None
    # The escalation rung's own bookkeeping, mirroring the breach
    # rung's: when the signed ``sla_escalation`` event first fired
    # (the dedupe marker for that rung) and its delivery ledger.
    sla_escalation_event_at: str | None = None
    sla_escalation_attempts: list[dict] = field(default_factory=list)
    # The Idempotency-Key the human decision was submitted with,
    # when the caller sent one (see service.approve / reject). A
    # repeat decision under the same key returns the recorded
    # decision instead of erroring or re-firing its effects; the
    # opposite decision under the same key is a conflict.
    decision_idempotency_key: str | None = None
    # ISO-8601 UTC timestamps, stamped by the service ("" until set —
    # older rows simply have none). The audit export reads them.
    created_at: str = ""  # when the analysis was recorded
    decided_at: str = ""  # when the human decision was recorded


def history_entry(record: ApprovalRecord) -> dict | None:
    """One prior-shipment history entry from a stored record, or None.

    Entries carry what the memory lookup matches and reports: the
    consignee (customer name, else a document's consignee field), the
    lane, and the exception the pipeline classified. Records stored
    without their shipment input (older rows) simply have no entry —
    memory degrades to "no history", never to an error.
    """
    shipment = record.shipment
    if not shipment:
        return None
    consignee = shipment.get("customer_name") or ""
    if not consignee:
        for document in shipment.get("documents", []):
            consignee = (document.get("fields") or {}).get("consignee") or ""
            if consignee:
                break
    classification = record.result.classification
    return {
        "shipment_id": record.result.shipment_id,
        "consignee": consignee,
        "carrier": shipment.get("carrier", ""),
        "origin": shipment.get("origin", ""),
        "destination": shipment.get("destination", ""),
        "lane": f"{shipment.get('origin', '')} -> {shipment.get('destination', '')}",
        "exception_type": classification.exception_type.value,
        "severity": classification.severity.value,
    }


def feedback_entry(record: ApprovalRecord) -> dict | None:
    """One reviewer-feedback entry from a decided record, or None.

    The feedback loop learns from oversight: a decision only teaches
    when the decider said *why*. Undecided records and reasonless
    decisions contribute nothing. Consignee/lane come from the same
    history entry the memory lookup uses, so feedback matches a new
    case exactly where memory would.
    """
    if record.approved:
        decision, reason, decider = "approved", record.approve_reason, record.approver
    elif record.rejected_by:
        decision, reason, decider = "rejected", record.reject_reason, record.rejected_by
    else:
        return None
    if not (reason or "").strip():
        return None
    entry = history_entry(record)
    if entry is None:
        return None
    return {
        "shipment_id": entry["shipment_id"],
        "consignee": entry["consignee"],
        "lane": entry["lane"],
        "decision": decision,
        "reason": reason,
        "decided_by": decider or "",
    }


def carrier_summary(entries: list[dict], carrier: str) -> dict:
    """One carrier's history over prior-shipment entries.

    Returns the total prior shipments with that carrier and the
    exception counts by type (``none`` results are shipments, not
    exceptions, so they count in the total only). Shared by the memory
    evidence in ``diagnosis.py`` and the ``carrier_history`` agent tool
    in ``tools_agent.py`` — one definition of the numbers, two surfaces.
    """
    type_counts: dict[str, int] = {}
    total = 0
    for entry in entries:
        if not carrier or entry.get("carrier") != carrier:
            continue
        total += 1
        exception = entry.get("exception_type")
        if exception and exception != "none":
            type_counts[exception] = type_counts.get(exception, 0) + 1
    return {
        "carrier": carrier,
        "carrier_count": total,
        "carrier_exception_count": sum(type_counts.values()),
        "carrier_type_counts": type_counts,
    }


def format_type_counts(type_counts: dict[str, int]) -> str:
    """``{"damage": 2, "delay": 1}`` -> ``"damage×2, delay×1"`` (count desc)."""
    ordered = sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{name}×{count}" for name, count in ordered)


# The Store protocol is declared in ports.py (the seam registry) and
# aliased here under its long-standing name, so existing imports keep
# working.
from .ports import Store as ApprovalStore  # noqa: E402,F401


def _record_to_dict(record: ApprovalRecord) -> dict:
    """The whole record as one JSON-ready dict — the Postgres payload
    shape (JSONB), and the shape the SQLite double's columns mirror."""
    return {
        "result": record.result.model_dump(mode="json"),
        "tenant_id": record.tenant_id,
        "shipment": record.shipment,
        "approver": record.approver,
        "approved": record.approved,
        "approve_reason": record.approve_reason,
        "rejected_by": record.rejected_by,
        "reject_reason": record.reject_reason,
        "dispatch_status": record.dispatch_status,
        "dispatch_attempts": record.dispatch_attempts,
        "idempotency_key": record.idempotency_key,
        "auth_key_id": record.auth_key_id,
        "sla_breach_event_at": record.sla_breach_event_at,
        "sla_dispatch_attempts": record.sla_dispatch_attempts,
        "sla_breach_age_seconds": record.sla_breach_age_seconds,
        "sla_escalation_event_at": record.sla_escalation_event_at,
        "sla_escalation_attempts": record.sla_escalation_attempts,
        "decision_idempotency_key": record.decision_idempotency_key,
        "created_at": record.created_at,
        "decided_at": record.decided_at,
    }


def _record_from_dict(data: dict) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult.model_validate(data["result"]),
        tenant_id=data.get("tenant_id") or DEFAULT_TENANT_ID,
        shipment=data.get("shipment"),
        approver=data.get("approver"),
        approved=data.get("approved", False),
        approve_reason=data.get("approve_reason", ""),
        rejected_by=data.get("rejected_by"),
        reject_reason=data.get("reject_reason", ""),
        dispatch_status=data.get("dispatch_status"),
        dispatch_attempts=data.get("dispatch_attempts") or [],
        idempotency_key=data.get("idempotency_key"),
        auth_key_id=data.get("auth_key_id"),
        sla_breach_event_at=data.get("sla_breach_event_at"),
        sla_dispatch_attempts=data.get("sla_dispatch_attempts") or [],
        sla_breach_age_seconds=data.get("sla_breach_age_seconds"),
        sla_escalation_event_at=data.get("sla_escalation_event_at"),
        sla_escalation_attempts=data.get("sla_escalation_attempts") or [],
        decision_idempotency_key=data.get("decision_idempotency_key"),
        created_at=data.get("created_at", ""),
        decided_at=data.get("decided_at", ""),
    )


class InMemoryStore:
    """Test double: records in a dict, gone when the process ends.

    A lock serialises access: the batch path (``analyze_batch``) runs
    analyses concurrently against one shared store, and each operation
    here is a read-modify or multi-step read that must not interleave.
    ``SQLiteStore`` gets the same guarantee by construction — it opens
    a fresh connection per call.
    """

    def __init__(self) -> None:
        # Keyed by (tenant_id, shipment_id): the store's identity is
        # the pair, so two tenants' SYN-1001s coexist without ever
        # shadowing each other.
        self._records: dict[tuple[str, str], ApprovalRecord] = {}
        self._worker_status: dict[str, dict] = {}
        self._summaries: dict[str, dict] = {}
        self._tenant_policies: dict[tuple[str, str], dict] = {}
        self._tenant_policy_history: dict[tuple[str, str], list[dict]] = {}
        self._lock = threading.Lock()

    # Worker status rows (see ports.Store): one summary per worker
    # name, replaced wholesale on each recorded sweep.
    def save_worker_status(self, worker: str, summary: dict) -> None:
        with self._lock:
            self._worker_status[worker] = summary

    def worker_status(self, worker: str) -> dict | None:
        with self._lock:
            return self._worker_status.get(worker)

    def all_worker_status(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._worker_status)

    # Summary rows (see ports.Store): one digest per key, replaced
    # wholesale on each composition.
    def save_summary(self, key: str, summary: dict) -> None:
        with self._lock:
            self._summaries[key] = summary

    def summary(self, key: str) -> dict | None:
        with self._lock:
            return self._summaries.get(key)

    # Tenant policy documents (see ports.Store): keyed by the
    # (tenant, policy) pair, like the records themselves.
    def save_tenant_policy(self, tenant_id: str, policy: dict) -> None:
        with self._lock:
            self._tenant_policies[(tenant_id, policy["policy_id"])] = dict(policy)

    def tenant_policy(self, tenant_id: str, policy_id: str) -> dict | None:
        with self._lock:
            policy = self._tenant_policies.get((tenant_id, policy_id))
            return dict(policy) if policy is not None else None

    def tenant_policies(self, tenant_id: str) -> list[dict]:
        with self._lock:
            return [
                dict(policy)
                for (tenant, _), policy in sorted(self._tenant_policies.items())
                if tenant == tenant_id
            ]

    def delete_tenant_policy(self, tenant_id: str, policy_id: str) -> bool:
        with self._lock:
            return self._tenant_policies.pop((tenant_id, policy_id), None) is not None

    # The tenant policy change ledger (see ports.Store): append-only
    # entries per (tenant, policy), read back oldest first.
    def record_tenant_policy_change(self, tenant_id: str, entry: dict) -> None:
        with self._lock:
            key = (tenant_id, entry["policy_id"])
            self._tenant_policy_history.setdefault(key, []).append(dict(entry))

    def tenant_policy_history(self, tenant_id: str, policy_id: str) -> list[dict]:
        with self._lock:
            return [
                dict(entry)
                for entry in self._tenant_policy_history.get(
                    (tenant_id, policy_id), []
                )
            ]

    def save(self, record: ApprovalRecord) -> None:
        with self._lock:
            key = (record.tenant_id, record.result.shipment_id)
            self._records[key] = record

    def get(
        self, shipment_id: str, tenant_id: str | None = None
    ) -> ApprovalRecord | None:
        with self._lock:
            if tenant_id is not None:
                return self._records.get((tenant_id, shipment_id))
            # Unscoped (operator) read: the newest record under this
            # id, whichever tenant owns it.
            for record in reversed(list(self._records.values())):
                if record.result.shipment_id == shipment_id:
                    return record
            return None

    def get_by_idempotency(
        self, key: str, shipment_id: str, tenant_id: str | None = None
    ) -> ApprovalRecord | None:
        with self._lock:
            if tenant_id is not None:
                candidates = [self._records.get((tenant_id, shipment_id))]
            else:
                candidates = [
                    record
                    for record in reversed(list(self._records.values()))
                    if record.result.shipment_id == shipment_id
                ]
        for record in candidates:
            if record is not None and record.idempotency_key == key:
                return record
        return None

    def records(self, tenant_id: str | None = None) -> list[ApprovalRecord]:
        """Stored records, newest first — one tenant's partition when
        ``tenant_id`` is given (metrics / audit read), else all."""
        with self._lock:
            records = list(reversed(list(self._records.values())))
        if tenant_id is None:
            return records
        return [record for record in records if record.tenant_id == tenant_id]

    def prior_shipments(
        self, exclude_shipment_id: str | None = None, tenant_id: str | None = None
    ) -> list[dict]:
        with self._lock:
            records = list(self._records.values())
        entries = []
        for record in reversed(records):
            if tenant_id is not None and record.tenant_id != tenant_id:
                continue
            if record.result.shipment_id == exclude_shipment_id:
                continue
            entry = history_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries

    def decision_feedback(
        self, exclude_shipment_id: str | None = None, tenant_id: str | None = None
    ) -> list[dict]:
        with self._lock:
            records = list(self._records.values())
        entries = []
        for record in reversed(records):
            if tenant_id is not None and record.tenant_id != tenant_id:
                continue
            if record.result.shipment_id == exclude_shipment_id:
                continue
            entry = feedback_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries


class SQLiteStore:
    """SQLite-backed store (stdlib only).

    One row per (tenant, shipment): the primary key is the pair, so
    two tenants' records with the same shipment id coexist.
    """

    # The columns every current row carries, in table order — the
    # rebuild below copies by name, so this list is the one place a
    # new column joins both the fresh-table and the migrated shapes.
    _COLUMNS = (
        "tenant_id, shipment_id, result_json, shipment_json, approver, "
        "approved, rejected_by, reject_reason, dispatch_status, "
        "approve_reason, created_at, decided_at, dispatch_attempts_json, "
        "idempotency_key, sla_breach_event_at, sla_dispatch_attempts_json, "
        "decision_idempotency_key, sla_breach_age_seconds, "
        "sla_escalation_event_at, sla_escalation_attempts_json, auth_key_id"
    )

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    shipment_id TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    approver TEXT,
                    approved INTEGER NOT NULL DEFAULT 0,
                    rejected_by TEXT,
                    reject_reason TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (tenant_id, shipment_id)
                )
                """
            )
            # Additive migration for databases created by earlier
            # versions: history lookups need the shipment input, and the
            # dispatch outcome needs a column. Existing rows keep NULLs
            # and simply contribute no history.
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(approvals)")
            }
            if "shipment_json" not in columns:
                conn.execute("ALTER TABLE approvals ADD COLUMN shipment_json TEXT")
            if "dispatch_status" not in columns:
                conn.execute("ALTER TABLE approvals ADD COLUMN dispatch_status TEXT")
            if "approve_reason" not in columns:
                conn.execute(
                    "ALTER TABLE approvals ADD COLUMN approve_reason TEXT NOT NULL DEFAULT ''"
                )
            if "created_at" not in columns:
                conn.execute(
                    "ALTER TABLE approvals ADD COLUMN created_at TEXT NOT NULL DEFAULT ''"
                )
            if "decided_at" not in columns:
                conn.execute(
                    "ALTER TABLE approvals ADD COLUMN decided_at TEXT NOT NULL DEFAULT ''"
                )
            if "dispatch_attempts_json" not in columns:
                conn.execute(
                    "ALTER TABLE approvals ADD COLUMN dispatch_attempts_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "idempotency_key" not in columns:
                conn.execute(
                    "ALTER TABLE approvals ADD COLUMN idempotency_key TEXT"
                )
            if "tenant_id" not in columns:
                conn.execute(
                    "ALTER TABLE approvals ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'"
                )
            if "sla_breach_event_at" not in columns:
                conn.execute(
                    "ALTER TABLE approvals ADD COLUMN sla_breach_event_at TEXT"
                )
            if "sla_dispatch_attempts_json" not in columns:
                conn.execute(
                    "ALTER TABLE approvals ADD COLUMN sla_dispatch_attempts_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "decision_idempotency_key" not in columns:
                conn.execute(
                    "ALTER TABLE approvals ADD COLUMN decision_idempotency_key TEXT"
                )
            if "sla_breach_age_seconds" not in columns:
                conn.execute(
                    "ALTER TABLE approvals ADD COLUMN sla_breach_age_seconds REAL"
                )
            if "sla_escalation_event_at" not in columns:
                conn.execute(
                    "ALTER TABLE approvals ADD COLUMN sla_escalation_event_at TEXT"
                )
            if "sla_escalation_attempts_json" not in columns:
                conn.execute(
                    "ALTER TABLE approvals ADD COLUMN sla_escalation_attempts_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "auth_key_id" not in columns:
                conn.execute(
                    "ALTER TABLE approvals ADD COLUMN auth_key_id TEXT"
                )
            # Tenancy changed the identity: a table created before it
            # keys rows by shipment_id alone, so two tenants' records
            # with the same id would shadow each other. Rebuild such a
            # table with the composite key, copying every row (their
            # tenant_id is the default the ALTER just stamped). The
            # identity columns in PRAGMA table_info carry pk > 0.
            pk_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(approvals)")
                if row["pk"]
            }
            if "tenant_id" not in pk_columns:
                conn.execute(
                    f"""
                    CREATE TABLE approvals_tenanted (
                        tenant_id TEXT NOT NULL DEFAULT 'default',
                        shipment_id TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        shipment_json TEXT,
                        approver TEXT,
                        approved INTEGER NOT NULL DEFAULT 0,
                        rejected_by TEXT,
                        reject_reason TEXT NOT NULL DEFAULT '',
                        dispatch_status TEXT,
                        approve_reason TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL DEFAULT '',
                        decided_at TEXT NOT NULL DEFAULT '',
                        dispatch_attempts_json TEXT NOT NULL DEFAULT '[]',
                        idempotency_key TEXT,
                        sla_breach_event_at TEXT,
                        sla_dispatch_attempts_json TEXT NOT NULL DEFAULT '[]',
                        decision_idempotency_key TEXT,
                        sla_breach_age_seconds REAL,
                        sla_escalation_event_at TEXT,
                        sla_escalation_attempts_json TEXT NOT NULL DEFAULT '[]',
                        auth_key_id TEXT,
                        PRIMARY KEY (tenant_id, shipment_id)
                    )
                    """
                )
                conn.execute(
                    f"INSERT OR IGNORE INTO approvals_tenanted ({self._COLUMNS}) "
                    f"SELECT {self._COLUMNS} FROM approvals"
                )
                conn.execute("DROP TABLE approvals")
                conn.execute("ALTER TABLE approvals_tenanted RENAME TO approvals")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    # Worker status rows (see ports.Store): a side table the store
    # owns its DDL for, like the approvals table above.
    def _ensure_worker_status_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS worker_status (
                worker TEXT PRIMARY KEY,
                summary_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT ''
            )
            """
        )

    def save_worker_status(self, worker: str, summary: dict) -> None:
        with self._connect() as conn:
            self._ensure_worker_status_table(conn)
            conn.execute(
                "INSERT OR REPLACE INTO worker_status "
                "(worker, summary_json, updated_at) VALUES (?, ?, ?)",
                (worker, json.dumps(summary), summary.get("updated_at", "")),
            )

    def worker_status(self, worker: str) -> dict | None:
        with self._connect() as conn:
            self._ensure_worker_status_table(conn)
            row = conn.execute(
                "SELECT summary_json FROM worker_status WHERE worker = ?",
                (worker,),
            ).fetchone()
        return json.loads(row["summary_json"]) if row else None

    def all_worker_status(self) -> dict[str, dict]:
        with self._connect() as conn:
            self._ensure_worker_status_table(conn)
            rows = conn.execute(
                "SELECT worker, summary_json FROM worker_status"
            ).fetchall()
        return {row["worker"]: json.loads(row["summary_json"]) for row in rows}

    # Summary rows (see ports.Store): a side table the store owns
    # its DDL for, like worker_status above.
    def _ensure_summaries_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                key TEXT PRIMARY KEY,
                summary_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT ''
            )
            """
        )

    def save_summary(self, key: str, summary: dict) -> None:
        with self._connect() as conn:
            self._ensure_summaries_table(conn)
            conn.execute(
                "INSERT OR REPLACE INTO summaries "
                "(key, summary_json, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(summary), summary.get("generated_at", "")),
            )

    def summary(self, key: str) -> dict | None:
        with self._connect() as conn:
            self._ensure_summaries_table(conn)
            row = conn.execute(
                "SELECT summary_json FROM summaries WHERE key = ?",
                (key,),
            ).fetchone()
        return json.loads(row["summary_json"]) if row else None

    # Tenant policy documents (see ports.Store): a side table the
    # store owns its DDL for, like worker_status above.
    def _ensure_tenant_policies_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tenant_policies (
                tenant_id TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (tenant_id, policy_id)
            )
            """
        )

    def save_tenant_policy(self, tenant_id: str, policy: dict) -> None:
        with self._connect() as conn:
            self._ensure_tenant_policies_table(conn)
            conn.execute(
                "INSERT OR REPLACE INTO tenant_policies "
                "(tenant_id, policy_id, payload_json) VALUES (?, ?, ?)",
                (tenant_id, policy["policy_id"], json.dumps(policy)),
            )

    def tenant_policy(self, tenant_id: str, policy_id: str) -> dict | None:
        with self._connect() as conn:
            self._ensure_tenant_policies_table(conn)
            row = conn.execute(
                "SELECT payload_json FROM tenant_policies "
                "WHERE tenant_id = ? AND policy_id = ?",
                (tenant_id, policy_id),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def tenant_policies(self, tenant_id: str) -> list[dict]:
        with self._connect() as conn:
            self._ensure_tenant_policies_table(conn)
            rows = conn.execute(
                "SELECT payload_json FROM tenant_policies "
                "WHERE tenant_id = ? ORDER BY policy_id",
                (tenant_id,),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def delete_tenant_policy(self, tenant_id: str, policy_id: str) -> bool:
        with self._connect() as conn:
            self._ensure_tenant_policies_table(conn)
            cursor = conn.execute(
                "DELETE FROM tenant_policies WHERE tenant_id = ? AND policy_id = ?",
                (tenant_id, policy_id),
            )
            return cursor.rowcount > 0

    def _ensure_tenant_policy_history_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tenant_policy_history (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )

    def record_tenant_policy_change(self, tenant_id: str, entry: dict) -> None:
        with self._connect() as conn:
            self._ensure_tenant_policy_history_table(conn)
            conn.execute(
                "INSERT INTO tenant_policy_history "
                "(tenant_id, policy_id, payload_json) VALUES (?, ?, ?)",
                (tenant_id, entry["policy_id"], json.dumps(entry)),
            )

    def tenant_policy_history(self, tenant_id: str, policy_id: str) -> list[dict]:
        with self._connect() as conn:
            self._ensure_tenant_policy_history_table(conn)
            rows = conn.execute(
                "SELECT payload_json FROM tenant_policy_history "
                "WHERE tenant_id = ? AND policy_id = ? ORDER BY seq",
                (tenant_id, policy_id),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def save(self, record: ApprovalRecord) -> None:


        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO approvals ({self._COLUMNS})
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.tenant_id,
                    record.result.shipment_id,
                    record.result.model_dump_json(),
                    json.dumps(record.shipment) if record.shipment else None,
                    record.approver,
                    1 if record.approved else 0,
                    record.rejected_by,
                    record.reject_reason,
                    record.dispatch_status,
                    record.approve_reason,
                    record.created_at,
                    record.decided_at,
                    json.dumps(record.dispatch_attempts),
                    record.idempotency_key,
                    record.sla_breach_event_at,
                    json.dumps(record.sla_dispatch_attempts),
                    record.decision_idempotency_key,
                    record.sla_breach_age_seconds,
                    record.sla_escalation_event_at,
                    json.dumps(record.sla_escalation_attempts),
                    record.auth_key_id,
                ),
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ApprovalRecord:


        keys = row.keys()
        shipment = None
        if "shipment_json" in keys and row["shipment_json"]:
            shipment = json.loads(row["shipment_json"])
        attempts: list[dict] = []
        if "dispatch_attempts_json" in keys and row["dispatch_attempts_json"]:
            try:
                attempts = json.loads(row["dispatch_attempts_json"]) or []
            except json.JSONDecodeError:
                attempts = []
        sla_attempts: list[dict] = []
        if "sla_dispatch_attempts_json" in keys and row["sla_dispatch_attempts_json"]:
            try:
                sla_attempts = json.loads(row["sla_dispatch_attempts_json"]) or []
            except json.JSONDecodeError:
                sla_attempts = []
        escalation_attempts: list[dict] = []
        if (
            "sla_escalation_attempts_json" in keys
            and row["sla_escalation_attempts_json"]
        ):
            try:
                escalation_attempts = (
                    json.loads(row["sla_escalation_attempts_json"]) or []
                )
            except json.JSONDecodeError:
                escalation_attempts = []
        return ApprovalRecord(
            result=AgentResult.model_validate_json(row["result_json"]),
            tenant_id=(
                row["tenant_id"]
                if "tenant_id" in keys and row["tenant_id"]
                else DEFAULT_TENANT_ID
            ),
            shipment=shipment,
            approver=row["approver"],
            approved=bool(row["approved"]),
            approve_reason=row["approve_reason"] if "approve_reason" in keys else "",
            rejected_by=row["rejected_by"],
            reject_reason=row["reject_reason"],
            dispatch_status=row["dispatch_status"] if "dispatch_status" in keys else None,
            dispatch_attempts=attempts,
            idempotency_key=row["idempotency_key"] if "idempotency_key" in keys else None,
            sla_breach_event_at=(
                row["sla_breach_event_at"] if "sla_breach_event_at" in keys else None
            ),
            sla_dispatch_attempts=sla_attempts,
            sla_breach_age_seconds=(
                row["sla_breach_age_seconds"]
                if "sla_breach_age_seconds" in keys
                else None
            ),
            sla_escalation_event_at=(
                row["sla_escalation_event_at"]
                if "sla_escalation_event_at" in keys
                else None
            ),
            sla_escalation_attempts=escalation_attempts,
            decision_idempotency_key=(
                row["decision_idempotency_key"]
                if "decision_idempotency_key" in keys
                else None
            ),
            auth_key_id=(
                row["auth_key_id"] if "auth_key_id" in keys else None
            ),
            created_at=row["created_at"] if "created_at" in keys else "",
            decided_at=row["decided_at"] if "decided_at" in keys else "",
        )

    def get(
        self, shipment_id: str, tenant_id: str | None = None
    ) -> ApprovalRecord | None:
        with self._connect() as conn:
            if tenant_id is not None:
                row = conn.execute(
                    "SELECT * FROM approvals WHERE tenant_id = ? AND shipment_id = ?",
                    (tenant_id, shipment_id),
                ).fetchone()
            else:  # unscoped (operator) read: the newest row under the id
                row = conn.execute(
                    "SELECT * FROM approvals WHERE shipment_id = ? "
                    "ORDER BY rowid DESC LIMIT 1",
                    (shipment_id,),
                ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_by_idempotency(
        self, key: str, shipment_id: str, tenant_id: str | None = None
    ) -> ApprovalRecord | None:
        with self._connect() as conn:
            if tenant_id is not None:
                row = conn.execute(
                    "SELECT * FROM approvals WHERE tenant_id = ? "
                    "AND shipment_id = ? AND idempotency_key = ?",
                    (tenant_id, shipment_id, key),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM approvals WHERE shipment_id = ? "
                    "AND idempotency_key = ? ORDER BY rowid DESC LIMIT 1",
                    (shipment_id, key),
                ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def records(self, tenant_id: str | None = None) -> list[ApprovalRecord]:
        """Stored records, newest first — one tenant's partition when
        ``tenant_id`` is given (metrics / audit read), else all."""
        with self._connect() as conn:
            if tenant_id is not None:
                rows = conn.execute(
                    "SELECT * FROM approvals WHERE tenant_id = ? ORDER BY rowid DESC",
                    (tenant_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM approvals ORDER BY rowid DESC"
                ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def prior_shipments(
        self, exclude_shipment_id: str | None = None, tenant_id: str | None = None
    ) -> list[dict]:
        entries = []
        for record in self.records(tenant_id):
            if record.result.shipment_id == exclude_shipment_id:
                continue
            entry = history_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries

    def decision_feedback(
        self, exclude_shipment_id: str | None = None, tenant_id: str | None = None
    ) -> list[dict]:
        entries = []
        for record in self.records(tenant_id):
            if record.result.shipment_id == exclude_shipment_id:
                continue
            entry = feedback_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries


class PostgresStore:
    """PostgreSQL-backed store — the production system of record.

    One short-lived psycopg connection per call (the same shape as
    the SQLite double), payload as JSONB, ordering by the identity
    ``seq`` column so history reads match insertion order. Schema is
    owned by the migrations (``db.ensure_migrated`` runs them before
    the first query); this class never issues DDL.
    """

    def __init__(self, url: str) -> None:
        from .db import ensure_migrated

        self._url = url
        ensure_migrated(url)

    def _connect(self):
        from .db import connect

        return connect(self._url)

    def save(self, record: ApprovalRecord) -> None:
        from psycopg.types.json import Jsonb

        decided = record.result.approval_status in ("approved", "rejected")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO approvals (tenant_id, shipment_id, payload, decided) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (tenant_id, shipment_id) DO UPDATE SET "
                "payload = EXCLUDED.payload, decided = EXCLUDED.decided, "
                "updated_at = now()",
                (
                    record.tenant_id,
                    record.result.shipment_id,
                    Jsonb(_record_to_dict(record)),
                    decided,
                ),
            )
            conn.commit()

    def get(
        self, shipment_id: str, tenant_id: str | None = None
    ) -> ApprovalRecord | None:
        with self._connect() as conn:
            if tenant_id is not None:
                row = conn.execute(
                    "SELECT payload FROM approvals "
                    "WHERE tenant_id = %s AND shipment_id = %s",
                    (tenant_id, shipment_id),
                ).fetchone()
            else:  # unscoped (operator) read: the newest row under the id
                row = conn.execute(
                    "SELECT payload FROM approvals WHERE shipment_id = %s "
                    "ORDER BY seq DESC LIMIT 1",
                    (shipment_id,),
                ).fetchone()
        return _record_from_dict(row[0]) if row else None

    def get_by_idempotency(
        self, key: str, shipment_id: str, tenant_id: str | None = None
    ) -> ApprovalRecord | None:
        with self._connect() as conn:
            if tenant_id is not None:
                row = conn.execute(
                    "SELECT payload FROM approvals WHERE tenant_id = %s "
                    "AND shipment_id = %s "
                    "AND payload ->> 'idempotency_key' = %s",
                    (tenant_id, shipment_id, key),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT payload FROM approvals WHERE shipment_id = %s "
                    "AND payload ->> 'idempotency_key' = %s "
                    "ORDER BY seq DESC LIMIT 1",
                    (shipment_id, key),
                ).fetchone()
        return _record_from_dict(row[0]) if row else None

    # Worker status rows (see ports.Store): the worker_status table
    # is migration 0006's; this class issues no DDL, as everywhere.
    def save_worker_status(self, worker: str, summary: dict) -> None:
        from psycopg.types.json import Jsonb

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO worker_status (worker, summary) VALUES (%s, %s) "
                "ON CONFLICT (worker) DO UPDATE SET "
                "summary = EXCLUDED.summary, updated_at = now()",
                (worker, Jsonb(summary)),
            )
            conn.commit()

    def worker_status(self, worker: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM worker_status WHERE worker = %s",
                (worker,),
            ).fetchone()
        return row[0] if row else None

    def all_worker_status(self) -> dict[str, dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT worker, summary FROM worker_status"
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    # Summary rows (see ports.Store): the summaries table is
    # migration 0008's; this class issues no DDL, as everywhere.
    def save_summary(self, key: str, summary: dict) -> None:
        from psycopg.types.json import Jsonb

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO summaries (key, summary) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET "
                "summary = EXCLUDED.summary, updated_at = now()",
                (key, Jsonb(summary)),
            )
            conn.commit()

    def summary(self, key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM summaries WHERE key = %s",
                (key,),
            ).fetchone()
        return row[0] if row else None

    # Tenant policy documents (see ports.Store): the tenant_policies
    # table is migration 0007's; this class issues no DDL, as
    # everywhere.
    def save_tenant_policy(self, tenant_id: str, policy: dict) -> None:
        from psycopg.types.json import Jsonb

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tenant_policies (tenant_id, policy_id, payload) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (tenant_id, policy_id) DO UPDATE SET "
                "payload = EXCLUDED.payload, updated_at = now()",
                (tenant_id, policy["policy_id"], Jsonb(policy)),
            )
            conn.commit()

    def tenant_policy(self, tenant_id: str, policy_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM tenant_policies "
                "WHERE tenant_id = %s AND policy_id = %s",
                (tenant_id, policy_id),
            ).fetchone()
        return row[0] if row else None

    def tenant_policies(self, tenant_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM tenant_policies "
                "WHERE tenant_id = %s ORDER BY policy_id",
                (tenant_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def delete_tenant_policy(self, tenant_id: str, policy_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM tenant_policies WHERE tenant_id = %s AND policy_id = %s",
                (tenant_id, policy_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    # The tenant policy change ledger (see ports.Store): the
    # tenant_policy_history table is migration 0009's; this class
    # issues no DDL, as everywhere.
    def record_tenant_policy_change(self, tenant_id: str, entry: dict) -> None:
        from psycopg.types.json import Jsonb

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tenant_policy_history (tenant_id, policy_id, payload) "
                "VALUES (%s, %s, %s)",
                (tenant_id, entry["policy_id"], Jsonb(entry)),
            )
            conn.commit()

    def tenant_policy_history(self, tenant_id: str, policy_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM tenant_policy_history "
                "WHERE tenant_id = %s AND policy_id = %s ORDER BY seq",
                (tenant_id, policy_id),
            ).fetchall()
        return [row[0] for row in rows]

    def _all_records(self, tenant_id: str | None = None) -> list[ApprovalRecord]:
        with self._connect() as conn:
            if tenant_id is not None:
                rows = conn.execute(
                    "SELECT payload FROM approvals WHERE tenant_id = %s "
                    "ORDER BY seq",
                    (tenant_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT payload FROM approvals ORDER BY seq"
                ).fetchall()
        return [_record_from_dict(row[0]) for row in rows]

    def records(self, tenant_id: str | None = None) -> list[ApprovalRecord]:
        """Stored records, newest first — one tenant's partition when
        ``tenant_id`` is given (metrics / audit read), else all."""
        return list(reversed(self._all_records(tenant_id)))

    def prior_shipments(
        self, exclude_shipment_id: str | None = None, tenant_id: str | None = None
    ) -> list[dict]:
        entries = []
        for record in reversed(self._all_records(tenant_id)):
            if record.result.shipment_id == exclude_shipment_id:
                continue
            entry = history_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries

    def decision_feedback(
        self, exclude_shipment_id: str | None = None, tenant_id: str | None = None
    ) -> list[dict]:
        entries = []
        for record in reversed(self._all_records(tenant_id)):
            if record.result.shipment_id == exclude_shipment_id:
                continue
            entry = feedback_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries


def default_store() -> ApprovalStore:
    """Resolve the store from the environment (see module docstring).

    ``DATABASE_URL`` set → :class:`PostgresStore` (the production
    system of record). ``STATE_DB_PATH`` set → the SQLite test double
    at that path (``:memory:`` → the in-memory double). Neither →
    :class:`InMemoryStore`: analyses and decisions live for the
    process only. There is deliberately no file-backed default any
    more — local files are not a runtime story (see
    ``docs/architecture.md``).
    """
    from .db import database_url

    url = database_url()
    if url:
        return PostgresStore(url)
    load_dotenv()
    configured = env_str("STATE_DB_PATH")
    if configured == ":memory:":
        return InMemoryStore()
    if configured:
        return SQLiteStore(configured)
    return InMemoryStore()
