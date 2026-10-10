-- 0004: multi-tenant partitioning of the approval store.
-- Every record belongs to a tenant; the store's identity becomes
-- the pair (tenant_id, shipment_id), so two tenants may both have
-- a SYN-1001 and neither can read the other's. Rows written before
-- tenancy belong to the default tenant (their payloads carry no
-- tenant_id) — the column default and the backfill agree on that.
ALTER TABLE approvals ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';
UPDATE approvals SET tenant_id = payload ->> 'tenant_id' WHERE payload ->> 'tenant_id' IS NOT NULL AND payload ->> 'tenant_id' <> '';
ALTER TABLE approvals DROP CONSTRAINT IF EXISTS approvals_pkey;
ALTER TABLE approvals ADD PRIMARY KEY (tenant_id, shipment_id);
CREATE INDEX IF NOT EXISTS approvals_tenant_idx ON approvals (tenant_id);
