"""Synthetic policy corpus used as the retrieval knowledge base.

In production this corpus would be the client's real SOPs, carrier claim
rules, and customer-communication policy, indexed from their document store.
Here it is a small, clearly synthetic in-repo corpus so the prototype runs
offline. ``data/sample/policies.json`` mirrors these entries.

**Per-tenant corpora.** One deployment serves many clients, and
clients do not share SOPs: :data:`TENANT_POLICIES` holds each
tenant's own documents, tagged with their ``tenant_id``, beside the
shared :data:`POLICIES` every tenant may cite. The retrievers scope
their corpus by tenant (see ``retriever.corpus_for_tenant``): a run
for tenant A retrieves the shared corpus plus A's documents — B's
documents are not ranked lower, they are *absent*, so they can never
surface in A's diagnosis, draft, or citations. ``data/sample/
tenant_policies.json`` mirrors the tenant entries.
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

# Each tenant's OWN documents, tagged with the tenant they belong to.
# They join the shared corpus for that tenant's runs only — the
# scoping lives in the retriever (retriever.corpus_for_tenant), so
# every retrieval mode (keyword / semantic / hybrid, and the pgvector
# table behind them) enforces the same partition.
TENANT_POLICIES: dict[str, list[dict[str, str]]] = {
    "acme": [
        {
            "policy_id": "SOP-ACME-01",
            "tenant_id": "acme",
            "title": "Reefer temperature excursion response (Acme)",
            "text": "When a reefer unit reports a temperature excursion above the setpoint on a cold-chain load, keep the trailer at the cross-dock, download the reefer telemetry log, and notify the Acme cold-chain duty manager before the freight is released. Record the excursion duration and the highest temperature reached in the load record.",
        },
    ],
    "globex": [
        {
            "policy_id": "SOP-GLOBEX-01",
            "tenant_id": "globex",
            "title": "High-value freight security protocol (Globex)",
            "text": "For Globex high-value loads, the trailer stays sealed and under camera watch at every stop, the security desk verifies the seal number at each handover, and any seal mismatch is treated as a security incident: the load does not move until the Globex security desk clears it.",
        },
    ],
}


def full_corpus() -> list[dict[str, str]]:
    """Every policy document the deployment knows: the shared corpus
    followed by every tenant's own tagged documents."""
    return [
        *POLICIES,
        *(policy for policies in TENANT_POLICIES.values() for policy in policies),
    ]


def policies_for_tenant(tenant_id: str | None) -> list[dict[str, str]]:
    """The corpus one tenant may see: the shared documents plus its
    own tagged ones. ``None`` (no tenant claimed) sees the shared
    corpus only — tenant documents never surface for an anonymous
    or another tenant's run."""
    if tenant_id is None:
        return list(POLICIES)
    return [
        *POLICIES,
        *TENANT_POLICIES.get(tenant_id, []),
    ]
