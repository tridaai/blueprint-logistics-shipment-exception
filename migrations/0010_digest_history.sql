-- 0010: digest snapshot history — the escalation digest, kept.
-- The summaries row (migration 0008) holds only the latest digest:
-- each sweep replaces it, so "what moved since the last picture?"
-- had no answer. This table is the append-only series beside it,
-- in the tenant_policy_history idiom (migration 0009): one row per
-- tenant per stored digest, carrying the snapshot's metadata —
-- per-stage counts, the awaiting shipments' ids and stages, the
-- window's firings by id, the oldest waiters, the rotation-window
-- state — plus a SHA-256 of the snapshot's canonical form.
-- Metadata and hashes, never shipment content: a digest snapshot
-- names ids and stages, exactly as the digest itself does. Rows
-- are pruned to a retention window by the store (a dated series,
-- not an audit ledger — the audit trail lives on the records).
CREATE TABLE IF NOT EXISTS digest_history (
    seq BIGINT GENERATED ALWAYS AS IDENTITY,
    tenant_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS digest_history_lookup
    ON digest_history (tenant_id, seq);
