-- 0007: tenant policy documents — the runtime-managed corpus.
-- A tenant's operators add, replace, and remove their own SOP
-- documents through the API (POST / DELETE /policies); the bundled
-- corpora in policies_data.py stay code, and these rows are the
-- slice of the corpus that changes without a deploy. One row per
-- (tenant, policy): the payload is the whole document
-- (policy_id / title / text / tenant_id / updated_at, plus the
-- object-store archive key when a store is configured). The
-- retriever merges these over the bundled corpus per run, scoped
-- to the owning tenant — a row here can never surface in another
-- tenant's retrieval.
CREATE TABLE IF NOT EXISTS tenant_policies (
    tenant_id TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, policy_id)
);
