-- 0001: extensions the production schema relies on.
-- pgvector serves the policy-corpus embeddings (migration 0003) from
-- the same PostgreSQL that holds the records and the checkpoints.
-- The compose stack ships the pgvector/pgvector image, where the
-- extension is available; on a managed Postgres it needs the
-- equivalent one-time enablement by the database owner.
CREATE EXTENSION IF NOT EXISTS vector;
