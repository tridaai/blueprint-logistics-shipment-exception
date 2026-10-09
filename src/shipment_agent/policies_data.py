"""Synthetic policy corpus used as the retrieval knowledge base.

In production this corpus would be the client's real SOPs, carrier claim
rules, and customer-communication policy, indexed from their document store.
Here it is a small, clearly synthetic in-repo corpus so the prototype runs
offline. ``data/sample/policies.json`` mirrors these entries.
"""

from __future__ import annotations

POLICIES: list[dict[str, str]] = [
    {
        "policy_id": "POL-DELAY-01",
        "title": "Proactive delay notification",
        "text": "When estimated delivery slips by four hours or more, notify the customer proactively with the revised estimate, the reason category, and the next update time. Do not wait for the customer to ask.",
    },
    {
        "policy_id": "POL-DMG-01",
        "title": "Damage documentation and claim window",
        "text": "For damaged freight, record the condition at delivery, photograph the packaging and goods, note the damage on the delivery receipt, and open the claim packet within one business day. Claims filed after the carrier window may be denied.",
    },
    {
        "policy_id": "POL-DOC-01",
        "title": "Document discrepancy hold",
        "text": "If the bill of lading and invoice disagree on quantity, weight, consignee, or SKU, place the shipment on document hold, notify the shipper of the exact fields in conflict, and do not release for final delivery until corrected documents are received.",
    },
    {
        "policy_id": "POL-APPT-01",
        "title": "Missed appointment rescheduling",
        "text": "When a delivery appointment is missed or refused at the dock, contact the receiving facility the same business day, secure the next available appointment, and inform the customer of the new window and any detention charges under review.",
    },
    {
        "policy_id": "POL-COMM-01",
        "title": "Customer communication standards",
        "text": "Customer updates must state what happened, what is being done, and when the next update will arrive. Never promise compensation, refunds, or guaranteed delivery times before the claim or delay review is complete.",
    },
    {
        "policy_id": "POL-CLAIM-01",
        "title": "Claim packet contents",
        "text": "A complete claim packet contains the bill of lading, the commercial invoice, the delivery receipt with condition notes, photographs of damage where applicable, and a written description of the exception and its timeline.",
    },
    {
        "policy_id": "POL-ESC-01",
        "title": "Escalation thresholds",
        "text": "Escalate to an operations lead when a delay exceeds 48 hours, damage is reported on a priority or critical service level shipment, or a document hold is unresolved after two business days.",
    },
    {
        "policy_id": "POL-NONE-01",
        "title": "Routine in-transit update",
        "text": "For shipments progressing normally, send a brief status update confirming the shipment is in transit, the current location if known, and the unchanged estimated delivery time.",
    },
]
