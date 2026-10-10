"""Webhook receiver-side helpers: verify, name, and dedupe events.

The sender's half of the signed-webhook story lives in
``service.py`` (``sign_webhook_body`` / ``verify_webhook_body`` and
the four dispatch paths). This module is the *receiver's* half,
written for the systems that consume the agent's events — and for
the ``shipment-agent verify-webhook`` CLI, which runs the same
checks over a captured payload:

- **Family.** Every event the agent sends belongs to one of five
  families: ``approval`` (the approved claim packet, the original
  output-routing payload — it predates the ``event`` field, so it
  is recognised by its shape), ``sla_breach`` and ``sla_escalation``
  (the queue SLA ladder's two rungs), ``worker_stale`` (the
  observer observing the observers), and ``corpus_changed`` (a
  tenant's operator changed the knowledge base — the event carries
  text hashes, never document text). One detection path names them
  all, so a receiver routes once instead of per-integration.
- **Event id.** A redelivery is byte-identical to the delivery it
  retries (the approval ledger re-sends the recorded packet; the
  SLA rungs fire once and are never re-composed), so the event's
  identity is the SHA-256 of its raw signed body, prefixed with
  the family: ``<family>:<hex>``. A receiver dedupes on that id —
  ack a duplicate exactly like the original (a sender that hears
  no ack retries), but apply its effect once.
- **Verdict.** :func:`inspect_webhook` runs the whole path over a
  captured body + signature: signature first (an unauthenticated
  body is never even parsed for routing), then family, then id.
"""

from __future__ import annotations

import hashlib
import json

from .service import verify_webhook_body

#: The event families a receiver can be sent, in send order.
EVENT_FAMILIES = (
    "approval",
    "sla_breach",
    "sla_escalation",
    "worker_stale",
    "corpus_changed",
)

_EVENT_FIELD_FAMILIES = (
    "sla_breach",
    "sla_escalation",
    "worker_stale",
    "corpus_changed",
)


def event_family(payload: dict) -> str | None:
    """Which family a parsed webhook body belongs to, or None.

    The SLA ladder's events and the staleness alert carry an
    explicit ``event`` field. The approval packet predates it: it
    is the payload with ``shipment_id`` + ``approval_status`` and
    no ``event`` field at all. Anything else is not an event this
    agent sends.
    """
    if not isinstance(payload, dict):
        return None
    event = payload.get("event")
    if event in _EVENT_FIELD_FAMILIES:
        return str(event)
    if event is None and "shipment_id" in payload and "approval_status" in payload:
        return "approval"
    return None


def event_id_for(body: bytes, family: str) -> str:
    """The dedupe identity of one delivered event.

    ``<family>:<sha256 hex of the raw body>``. Byte-identical
    redeliveries (the only kind this agent produces — see the
    module docstring) share an id; two different events never do,
    short of a hash collision.
    """
    return f"{family}:{hashlib.sha256(body).hexdigest()}"


def inspect_webhook(body: bytes, signature: str | None, secret: str) -> dict:
    """Run the receiver's checks over one captured delivery.

    Returns a verdict dict: ``{"verified": bool, "family": str |
    None, "event_id": str | None, "error": str | None}``. The
    signature is checked first and the body is only parsed once it
    authenticates — a receiver must never route on unauthenticated
    content. ``error`` names the failing check for the operator
    reading the CLI output; it is deliberately coarse (a sender
    probing the receiver learns nothing from it beyond "no").
    """
    if not verify_webhook_body(body, signature, secret):
        return {
            "verified": False,
            "family": None,
            "event_id": None,
            "error": "signature verification failed",
        }
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {
            "verified": True,
            "family": None,
            "event_id": None,
            "error": "body is not valid JSON",
        }
    family = event_family(payload)
    if family is None:
        return {
            "verified": True,
            "family": None,
            "event_id": None,
            "error": "not a recognised event family",
        }
    return {
        "verified": True,
        "family": family,
        "event_id": event_id_for(body, family),
        "error": None,
    }
