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
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import env_str, load_dotenv
from .schemas import AgentResult


@dataclass
class ApprovalRecord:
    result: AgentResult
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
        "shipment": record.shipment,
        "approver": record.approver,
        "approved": record.approved,
        "approve_reason": record.approve_reason,
        "rejected_by": record.rejected_by,
        "reject_reason": record.reject_reason,
        "dispatch_status": record.dispatch_status,
        "dispatch_attempts": record.dispatch_attempts,
        "idempotency_key": record.idempotency_key,
        "created_at": record.created_at,
        "decided_at": record.decided_at,
    }


def _record_from_dict(data: dict) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult.model_validate(data["result"]),
        shipment=data.get("shipment"),
        approver=data.get("approver"),
        approved=data.get("approved", False),
        approve_reason=data.get("approve_reason", ""),
        rejected_by=data.get("rejected_by"),
        reject_reason=data.get("reject_reason", ""),
        dispatch_status=data.get("dispatch_status"),
        dispatch_attempts=data.get("dispatch_attempts") or [],
        idempotency_key=data.get("idempotency_key"),
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
        self._records: dict[str, ApprovalRecord] = {}
        self._lock = threading.Lock()

    def save(self, record: ApprovalRecord) -> None:
        with self._lock:
            self._records[record.result.shipment_id] = record

    def get(self, shipment_id: str) -> ApprovalRecord | None:
        with self._lock:
            return self._records.get(shipment_id)

    def get_by_idempotency(
        self, key: str, shipment_id: str
    ) -> ApprovalRecord | None:
        with self._lock:
            record = self._records.get(shipment_id)
        if record is not None and record.idempotency_key == key:
            return record
        return None

    def records(self) -> list[ApprovalRecord]:
        """Every stored record, newest first (metrics / audit read)."""
        with self._lock:
            return list(reversed(list(self._records.values())))

    def prior_shipments(self, exclude_shipment_id: str | None = None) -> list[dict]:
        with self._lock:
            records = list(self._records.values())
        entries = []
        for record in reversed(records):
            if record.result.shipment_id == exclude_shipment_id:
                continue
            entry = history_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries

    def decision_feedback(self, exclude_shipment_id: str | None = None) -> list[dict]:
        with self._lock:
            records = list(self._records.values())
        entries = []
        for record in reversed(records):
            if record.result.shipment_id == exclude_shipment_id:
                continue
            entry = feedback_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries


class SQLiteStore:
    """SQLite-backed store (stdlib only). One row per shipment."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    shipment_id TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    approver TEXT,
                    approved INTEGER NOT NULL DEFAULT 0,
                    rejected_by TEXT,
                    reject_reason TEXT NOT NULL DEFAULT ''
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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, record: ApprovalRecord) -> None:


        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO approvals
                    (shipment_id, result_json, shipment_json, approver, approved,
                     rejected_by, reject_reason, dispatch_status, approve_reason,
                     created_at, decided_at, dispatch_attempts_json, idempotency_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
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
        return ApprovalRecord(
            result=AgentResult.model_validate_json(row["result_json"]),
            shipment=shipment,
            approver=row["approver"],
            approved=bool(row["approved"]),
            approve_reason=row["approve_reason"] if "approve_reason" in keys else "",
            rejected_by=row["rejected_by"],
            reject_reason=row["reject_reason"],
            dispatch_status=row["dispatch_status"] if "dispatch_status" in keys else None,
            dispatch_attempts=attempts,
            idempotency_key=row["idempotency_key"] if "idempotency_key" in keys else None,
            created_at=row["created_at"] if "created_at" in keys else "",
            decided_at=row["decided_at"] if "decided_at" in keys else "",
        )

    def get(self, shipment_id: str) -> ApprovalRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approvals WHERE shipment_id = ?", (shipment_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_by_idempotency(
        self, key: str, shipment_id: str
    ) -> ApprovalRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approvals WHERE shipment_id = ? AND idempotency_key = ?",
                (shipment_id, key),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def records(self) -> list[ApprovalRecord]:
        """Every stored record, newest first (metrics / audit read)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM approvals ORDER BY rowid DESC"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def prior_shipments(self, exclude_shipment_id: str | None = None) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM approvals ORDER BY rowid DESC"
            ).fetchall()
        entries = []
        for row in rows:
            record = self._row_to_record(row)
            if record.result.shipment_id == exclude_shipment_id:
                continue
            entry = history_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries

    def decision_feedback(self, exclude_shipment_id: str | None = None) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM approvals ORDER BY rowid DESC"
            ).fetchall()
        entries = []
        for row in rows:
            record = self._row_to_record(row)
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
                "INSERT INTO approvals (shipment_id, payload, decided) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (shipment_id) DO UPDATE SET "
                "payload = EXCLUDED.payload, decided = EXCLUDED.decided, "
                "updated_at = now()",
                (
                    record.result.shipment_id,
                    Jsonb(_record_to_dict(record)),
                    decided,
                ),
            )
            conn.commit()

    def get(self, shipment_id: str) -> ApprovalRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM approvals WHERE shipment_id = %s",
                (shipment_id,),
            ).fetchone()
        return _record_from_dict(row[0]) if row else None

    def get_by_idempotency(
        self, key: str, shipment_id: str
    ) -> ApprovalRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM approvals WHERE shipment_id = %s "
                "AND payload ->> 'idempotency_key' = %s",
                (shipment_id, key),
            ).fetchone()
        return _record_from_dict(row[0]) if row else None

    def _all_records(self) -> list[ApprovalRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM approvals ORDER BY seq"
            ).fetchall()
        return [_record_from_dict(row[0]) for row in rows]

    def records(self) -> list[ApprovalRecord]:
        """Every stored record, newest first (metrics / audit read)."""
        return list(reversed(self._all_records()))

    def prior_shipments(self, exclude_shipment_id: str | None = None) -> list[dict]:
        entries = []
        for record in reversed(self._all_records()):
            if record.result.shipment_id == exclude_shipment_id:
                continue
            entry = history_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries

    def decision_feedback(self, exclude_shipment_id: str | None = None) -> list[dict]:
        entries = []
        for record in reversed(self._all_records()):
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
