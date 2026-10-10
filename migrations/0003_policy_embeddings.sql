-- 0003: policy corpus embeddings, served by pgvector.
-- One row per (policy, embedding model): switching models never
-- mixes vector spaces. content_hash detects corpus edits so stale
-- embeddings are re-embedded on next retrieval, not served.
-- Deliberately no ANN index: the corpus is small and a sequential
-- cosine scan is exact; a production corpus at scale would pin the
-- embedding dimension and add an HNSW index (see architecture.md).
CREATE TABLE IF NOT EXISTS policy_embeddings (
    policy_id TEXT NOT NULL,
    model TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (policy_id, model)
);
