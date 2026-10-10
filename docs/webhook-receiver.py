#!/usr/bin/env python3
"""Reference webhook receiver for the Shipment Exception Agent.

A runnable, dependency-free (stdlib only) receiver for every event
the agent sends, through ONE code path:

    verify signature -> parse -> name the family -> dedupe -> ack

Run it, point the agent at it, and watch real events land:

    WEBHOOK_RECEIVER_SECRET=<the same secret as ACTION_WEBHOOK_SECRET> \
        python3 docs/webhook-receiver.py --port 8787

    # agent side:
    ACTION_WEBHOOK_URL=http://localhost:8787/webhooks \
    ACTION_WEBHOOK_SECRET=<the same secret> \
    SLA_BREACH_WEBHOOK=on            # for the SLA + staleness families

The four families (route on the returned ``family``):

- ``approval``       — the approved claim packet (ACTION_WEBHOOK_URL).
                       The original output-routing payload: it carries
                       ``shipment_id`` + ``approval_status`` and no
                       ``event`` field, so it is recognised by shape.
- ``sla_breach``     — a queue item blew its severity's SLA budget.
- ``sla_escalation`` — a reported breach kept aging past the ladder's
                       second threshold.
- ``worker_stale``   — a background worker stopped reporting.

The rules this file exists to demonstrate (receivers keep getting
them wrong, which is why it ships):

1. **Verify over the RAW body.** The signature is
   ``X-Trida-Signature: sha256=<HMAC-SHA256 hex>`` keyed by the
   shared secret over the exact bytes received. Re-serialising the
   parsed JSON and verifying that instead fails intermittently —
   key order and whitespace are part of the signed content.
   Compare in constant time (``hmac.compare_digest``).
2. **Verify before you parse for routing.** An unauthenticated
   body is answered 401 and never routed on.
3. **Dedupe on the event id, ack duplicates.** A redelivery is
   byte-identical to the original, so the event id is
   ``<family>:<sha256 of the raw body>``. A duplicate gets the
   same 200 ack as the first delivery (a sender that hears no ack
   retries — acking is what stops the retries) but its effect is
   applied once. Seen ids persist in an append-only JSONL log
   (``--log``, default ``webhook-events.jsonl``) so a receiver
   restart does not reopen the dedupe window.
4. **Ack shape.** 200 with ``{"received": true, "family": ...,
   "event_id": ..., "duplicate": bool}``. Non-2xx answers are
   recorded as failed deliveries in the agent's ledgers, so only
   answer an error when the event really was not accepted.

Checking a captured payload offline instead of running a server?
``shipment-agent verify-webhook <payload.json> --signature sha256=...``
runs the same verification + family + event-id path from the CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EVENT_FAMILIES = ("approval", "sla_breach", "sla_escalation", "worker_stale")
_EVENT_FIELD_FAMILIES = ("sla_breach", "sla_escalation", "worker_stale")


def verify_signature(body: bytes, signature: str | None, secret: str) -> bool:
    """Does ``signature`` authenticate ``body`` under ``secret``?

    The receiver's half of the signing contract: recompute the
    HMAC over the raw bytes and compare in constant time."""
    if not signature or not secret:
        return False
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={digest}", signature.strip())


def event_family(payload: dict) -> str | None:
    """Which family a parsed body belongs to (see module docstring)."""
    if not isinstance(payload, dict):
        return None
    event = payload.get("event")
    if event in _EVENT_FIELD_FAMILIES:
        return str(event)
    if event is None and "shipment_id" in payload and "approval_status" in payload:
        return "approval"
    return None


def event_id_for(body: bytes, family: str) -> str:
    """The dedupe identity: ``<family>:<sha256 of the raw body>``."""
    return f"{family}:{hashlib.sha256(body).hexdigest()}"


class EventLog:
    """Append-only JSONL record of seen events; the dedupe store.

    Reloaded at startup, so the dedupe window survives restarts.
    One line per accepted delivery: ``{"event_id", "family",
    "received_at", "duplicate"}`` — received facts only, never the
    payload itself (the payload is the sender's record to keep).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("event_id"):
                    self._seen.add(entry["event_id"])

    def record(self, event_id: str, family: str) -> bool:
        """Record one accepted delivery. True when it is a duplicate."""
        with self._lock:
            duplicate = event_id in self._seen
            self._seen.add(event_id)
            entry = {
                "event_id": event_id,
                "family": family,
                "received_at": datetime.now(timezone.utc).isoformat(),
                "duplicate": duplicate,
            }
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            return duplicate


def make_handler(secret: str, log: EventLog):
    """The request handler: one code path for all four families."""

    class WebhookHandler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            # 1. Verify over the raw bytes, before any routing.
            signature = self.headers.get("X-Trida-Signature")
            if not verify_signature(body, signature, secret):
                return self._answer(401, {"error": "signature verification failed"})
            # 2. Parse and name the family.
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._answer(400, {"error": "body is not valid JSON"})
            family = event_family(payload)
            if family is None:
                return self._answer(400, {"error": "not a recognised event family"})
            # 3. Dedupe on the event id; ack duplicates like originals.
            event_id = event_id_for(body, family)
            duplicate = log.record(event_id, family)
            print(
                f"{family} {event_id[:40]}… "
                f"{'duplicate (acked, not re-applied)' if duplicate else 'accepted'}",
                flush=True,
            )
            # 4. Your system of record applies the event HERE, once,
            #    when `duplicate` is False. This reference receiver
            #    only logs — the seam is the point.
            return self._answer(
                200,
                {
                    "received": True,
                    "family": family,
                    "event_id": event_id,
                    "duplicate": duplicate,
                },
            )

        def do_GET(self):  # noqa: N802 - a liveness answer for operators
            if self.path == "/health":
                return self._answer(200, {"status": "ok"})
            return self._answer(404, {"error": "not found"})

        def _answer(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # the prints above are the log
            pass

    return WebhookHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reference receiver for the Shipment Exception Agent's webhooks"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--secret",
        default=os.environ.get("WEBHOOK_RECEIVER_SECRET", ""),
        help="The shared secret (default: $WEBHOOK_RECEIVER_SECRET). "
        "Must match the agent's ACTION_WEBHOOK_SECRET.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("webhook-events.jsonl"),
        help="Append-only event log / dedupe store (default: webhook-events.jsonl)",
    )
    args = parser.parse_args(argv)
    if not args.secret:
        print(
            "webhook-receiver: no secret configured — set "
            "WEBHOOK_RECEIVER_SECRET (or pass --secret) to the same "
            "value as the agent's ACTION_WEBHOOK_SECRET. A receiver "
            "that cannot verify signatures is not a receiver.",
            file=sys.stderr,
        )
        return 2
    log = EventLog(args.log)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(args.secret, log)
    )
    print(
        f"webhook-receiver: listening on http://{args.host}:{args.port} "
        f"(families: {', '.join(EVENT_FAMILIES)}; log: {args.log})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("webhook-receiver: stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
