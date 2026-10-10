-- 0005: the tenant axis on the policy-corpus embeddings.
-- Policy documents are tagged: shared documents serve every tenant,
-- a tenant's own SOPs serve that tenant alone. The pgvector table
-- gains the tag so the retriever's query filters on it (shared rows
-- are NULL; see retriever._retrieve_pgvector). Rows written before
-- this migration are all shared-corpus rows, which is exactly what
-- the NULL default says. Policy ids stay unique across corpora, so
-- the (policy_id, model) key is unchanged.
ALTER TABLE policy_embeddings ADD COLUMN IF NOT EXISTS tenant_id TEXT;
