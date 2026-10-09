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

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import env_str, load_dotenv
from .schemas import AgentResult


@dataclass
class ApprovalRecord:
    result: AgentResult
    approver: str | None = None
    approved: bool = False
    rejected_by: str | None = None
    reject_reason: str = ""


class ApprovalStore(Protocol):
    def save(self, record: ApprovalRecord) -> None: ...
    def get(self, shipment_id: str) -> ApprovalRecord | None: ...


class InMemoryStore:
    """Test double: records in a dict, gone when the process ends."""

    def __init__(self) -> None:
        self._records: dict[str, ApprovalRecord] = {}

    def save(self, record: ApprovalRecord) -> None:
        self._records[record.result.shipment_id] = record

    def get(self, shipment_id: str) -> ApprovalRecord | None:
        return self._records.get(shipment_id)


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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, record: ApprovalRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO approvals
                    (shipment_id, result_json, approver, approved, rejected_by, reject_reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.result.shipment_id,
                    record.result.model_dump_json(),
                    record.approver,
                    1 if record.approved else 0,
                    record.rejected_by,
                    record.reject_reason,
                ),
            )

    def get(self, shipment_id: str) -> ApprovalRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approvals WHERE shipment_id = ?", (shipment_id,)
            ).fetchone()
        if row is None:
            return None
        return ApprovalRecord(
            result=AgentResult.model_validate_json(row["result_json"]),
            approver=row["approver"],
            approved=bool(row["approved"]),
            rejected_by=row["rejected_by"],
            reject_reason=row["reject_reason"],
        )


def default_store() -> ApprovalStore:
    """Resolve the store from the environment (see module docstring)."""
    load_dotenv()
    configured = env_str("STATE_DB_PATH")
    if configured == ":memory:":
        return InMemoryStore()
    if configured:
        return SQLiteStore(configured)
    return SQLiteStore(Path(__file__).resolve().parents[2] / ".data" / "state.db")
