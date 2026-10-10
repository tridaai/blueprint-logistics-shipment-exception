"""Audit trail: the decisions record, exportable.

``GET /audit/export`` serves what an auditor (or an ops lead doing a
review) actually asks for: one row per analysed shipment — what the
agent concluded, who decided, why, and when — as JSON or CSV. The
rows are a projection of the approval store, the system of record;
nothing here is a second copy that can drift. Undecided analyses are
included with an empty decision: the trail shows the queue as well
as the outcomes.
"""

from __future__ import annotations

import csv
import io

from .store import ApprovalRecord

AUDIT_FIELDS = (
    "shipment_id",
    "exception_type",
    "severity",
    "approval_status",
    "decision",
    "decided_by",
    "decision_reason",
    "guardrails_passed",
    "dispatch_status",
    "created_at",
    "decided_at",
)


def audit_rows(records: list[ApprovalRecord]) -> list[dict]:
    """One audit row per record, newest first (the store's order)."""
    rows: list[dict] = []
    for record in records:
        result = record.result
        if result.approval_status == "approved":
            decision, decided_by, reason = (
                "approved",
                record.approver or result.decided_by or "",
                record.approve_reason,
            )
        elif result.approval_status == "rejected":
            decision, decided_by, reason = (
                "rejected",
                record.rejected_by or result.decided_by or "",
                record.reject_reason,
            )
        else:
            decision, decided_by, reason = "", "", ""
        rows.append(
            {
                "shipment_id": result.shipment_id,
                "exception_type": result.classification.exception_type.value,
                "severity": result.classification.severity.value,
                "approval_status": result.approval_status,
                "decision": decision,
                "decided_by": decided_by,
                "decision_reason": reason,
                "guardrails_passed": result.validation.passed,
                "dispatch_status": record.dispatch_status or "",
                "created_at": record.created_at,
                "decided_at": record.decided_at,
            }
        )
    return rows


def render_audit_csv(rows: list[dict]) -> str:
    """The audit rows as CSV (header + one line per row)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(AUDIT_FIELDS))
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()
