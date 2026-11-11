-- Reference SQL: index definitions (§4.7)
-- RAG_SPEC 1.2.0.
-- StoreCapabilities:
--   supports_vector_search=True
--   supports_keyword_fts=True
--   supports_sequential_boundary=True

-- 1. HNSW on the embedding column. Rows stored without a vector (embedding IS NULL) are skipped.
--    With pgvector >= 0.8 the search RPCs enable iterative scans, so filtered queries still fill k rows.
CREATE INDEX IF NOT EXISTS idx_passages_embedding_hnsw
ON passages
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 2. GIN on the full-text tokens
CREATE INDEX IF NOT EXISTS idx_passages_fts_tokens_gin
ON passages
USING gin (fts_search_tokens);

-- 3. B-tree for the owner and document scope every RPC filters on first (§5.6, §6.4.2)
CREATE INDEX IF NOT EXISTS idx_passages_owner_doc
ON passages (tenant_or_user_id, doc_id, is_active);

-- 4. B-tree on (doc_id, char_start) for boundaries, local windows and full-scope order
CREATE INDEX IF NOT EXISTS idx_passages_doc_char_start
ON passages (doc_id, char_start);

-- 5. Partial index for the summary layer (fetch_parent_summaries)
CREATE INDEX IF NOT EXISTS idx_passages_summaries
ON passages (tenant_or_user_id, doc_id, char_end)
WHERE is_summary = TRUE;
