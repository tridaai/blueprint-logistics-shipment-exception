-- 0008: summary rows — named, deployment-wide digests.
-- A background process composes a digest and the API serves the
-- stored row: today the queue's escalation digest (breaches by
-- stage, the last 24h of ladder firings, oldest waiters per
-- severity per tenant, stale workers, open key-rotation windows),
-- written by the SLA sweep under the key 'queue_digest' and read
-- by GET /queue/digest. One row per key, replaced on each
-- composition. Digests carry counts and ids, never shipment
-- content.
CREATE TABLE IF NOT EXISTS summaries (
    key TEXT PRIMARY KEY,
    summary JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
