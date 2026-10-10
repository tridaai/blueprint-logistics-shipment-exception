-- 0006: worker status rows — the background processes' heartbeat.
-- The dispatch-retry worker and the SLA sweep are separate
-- processes from the API; without a shared record of their runs, a
-- worker that stopped sweeping is indistinguishable from one with
-- nothing to do. Each worker folds every sweep into its row here
-- (last sweep time, sweep count, cumulative outcomes overall and
-- per tenant), and the API's /metrics reads the rows. Summaries
-- carry counts, never shipment content.
CREATE TABLE IF NOT EXISTS worker_status (
    worker TEXT PRIMARY KEY,
    summary JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
