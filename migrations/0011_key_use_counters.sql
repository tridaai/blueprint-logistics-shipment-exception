-- 0011: key-use counters — the evidence a key rotation closes on.
-- Rotation readiness used to read stored analyses only: a tenant's
-- outgoing key used solely for reads (the queue, the corpus, this
-- very status view) left no record, so the "the old key has gone
-- quiet" verdict could overstate. This table holds one counter row
-- per (tenant, key id): how many authenticated requests that key
-- generation has served, when the first and last ones were, and the
-- same facts scoped to the current rotation window. Key ids name a
-- generation (`acme:previous`, `shared`) — the secret itself is
-- never stored, here or anywhere. Not an audit trail: one row per
-- key, updated in place, so the table stays the size of the key
-- inventory no matter how busy the API gets.
CREATE TABLE IF NOT EXISTS key_use_counters (
    tenant_id TEXT NOT NULL,
    key_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, key_id)
);
