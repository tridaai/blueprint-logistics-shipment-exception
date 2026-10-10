-- 0009: tenant policy change ledger — who changed the corpus, when.
-- Tenant policy documents (migration 0007) are mutable rows: an
-- operator who replaces an SOP otherwise leaves no trace of what
-- changed, when, or under whose key. This table is the append-only
-- ledger beside the documents: one row per add / replace / remove,
-- carrying the actor's key id (never the secret), the change time,
-- the SHA-256 of the text before and after (hashes, never the text
-- itself — the current document lives in tenant_policies, and a
-- removed document's content is gone by the operator's intent), and
-- the corpus_changed webhook's delivery outcome. `seq` preserves
-- insertion order, so a document's history reads oldest first.
CREATE TABLE IF NOT EXISTS tenant_policy_history (
    seq BIGINT GENERATED ALWAYS AS IDENTITY,
    tenant_id TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tenant_policy_history_lookup
    ON tenant_policy_history (tenant_id, policy_id, seq);
