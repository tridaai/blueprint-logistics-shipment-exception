"""Approval persistence: where analysis results and human decisions live.

Two implementations behind one small interface:

- ``SQLiteStore`` — the real one. Stdlib ``sqlite3``, one local file.
  Results (as JSON) plus the decision columns (approver, approved,
  rejected_by, reject_reason) survive process restarts: an approval
  queue that evaporates when the server restarts is not a product.
- ``InMemoryStore`` — the test double. Same interface, no file; this
  was the only store before v2 and remains what most unit tests use.

Selection (``default_store``): ``STATE_DB_PATH`` env var — a file path
for SQLite, or ``:memory:`` for the in-memory store. Unset, it defaults
to ``<repo>/.data/state.db`` (git-ignored). Re-analyzing a shipment
replaces its record and resets the decision, matching the service's
long-standing behaviour.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import env_str, load_dotenv
from .schemas import AgentResult


@dataclass
class ApprovalRecord:
    result: AgentResult
    shipment: dict | None = None  # the analysed ShipmentInput, for history lookups
    approver: str | None = None
    approved: bool = False
    rejected_by: str | None = None
    reject_reason: str = ""
    dispatch_status: str | None = None  # outbound webhook outcome, when configured


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
        "origin": shipment.get("origin", ""),
        "destination": shipment.get("destination", ""),
        "lane": f"{shipment.get('origin', '')} -> {shipment.get('destination', '')}",
        "exception_type": classification.exception_type.value,
        "severity": classification.severity.value,
    }


class ApprovalStore(Protocol):
    def save(self, record: ApprovalRecord) -> None: ...
    def get(self, shipment_id: str) -> ApprovalRecord | None: ...
    def prior_shipments(self, exclude_shipment_id: str | None = None) -> list[dict]: ...


class InMemoryStore:
    """Test double: records in a dict, gone when the process ends."""

    def __init__(self) -> None:
        self._records: dict[str, ApprovalRecord] = {}

    def save(self, record: ApprovalRecord) -> None:
        self._records[record.result.shipment_id] = record

    def get(self, shipment_id: str) -> ApprovalRecord | None:
        return self._records.get(shipment_id)

    def prior_shipments(self, exclude_shipment_id: str | None = None) -> list[dict]:
        entries = []
        for record in reversed(list(self._records.values())):
            if record.result.shipment_id == exclude_shipment_id:
                continue
            entry = history_entry(record)
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
                     rejected_by, reject_reason, dispatch_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ApprovalRecord:


        keys = row.keys()
        shipment = None
        if "shipment_json" in keys and row["shipment_json"]:
            shipment = json.loads(row["shipment_json"])
        return ApprovalRecord(
            result=AgentResult.model_validate_json(row["result_json"]),
            shipment=shipment,
            approver=row["approver"],
            approved=bool(row["approved"]),
            rejected_by=row["rejected_by"],
            reject_reason=row["reject_reason"],
            dispatch_status=row["dispatch_status"] if "dispatch_status" in keys else None,
        )

    def get(self, shipment_id: str) -> ApprovalRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approvals WHERE shipment_id = ?", (shipment_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

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


def default_store() -> ApprovalStore:
    """Resolve the store from the environment (see module docstring)."""
    load_dotenv()
    configured = env_str("STATE_DB_PATH")
    if configured == ":memory:":
        return InMemoryStore()
    if configured:
        return SQLiteStore(configured)
    return SQLiteStore(Path(__file__).resolve().parents[2] / ".data" / "state.db")
