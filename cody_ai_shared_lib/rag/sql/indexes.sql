-- MIGRATION CHECKLIST (RAG_SPEC 1.2.0 §4.7): check your own functions line by line against these.
--   [ ] Upsert never deletes: only the rows in the write are inserted or updated; deletion is a separate RPC.
--   [ ] Owner: every function takes the owner (tenant_or_user_id), raises when it is missing, filters on it,
--       and refuses a write into a document or row another owner holds.
--   [ ] search_path pinned on every SECURITY DEFINER function: `public, extensions, pg_temp`
--       (Supabase installs pgvector in `extensions`; pg_temp last stops temp-schema shadowing).
--   [ ] Summaries admitted only when char_end <= boundary; ranked searches read passages only.
--   [ ] Scan mode: pgvector >= 0.8 iterative index scans, or an exact distance sort when every query is
--       limited to one document (then no vector index is needed).
--   [ ] Text by reference: slice chunk text with chunk-relative offsets (char_start - chunk.char_start).
--   [ ] Fingerprints stored exactly as the engine sends them; never computed or rebuilt in SQL.

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
