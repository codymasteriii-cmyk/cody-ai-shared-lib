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

-- Reference SQL: CJK Full-Text Search Options (§4.7, §6.1, AC-RAG-67)
-- StoreCapabilities:
--   supports_keyword_fts=True
--   uses_analyzer_tokens=True

-- Option A: simple-config tsvector over TextAnalyzer search_tokens (Default on Supabase)
-- In this option, TextAnalyzer produces search_tokens at ingestion (e.g. '["主角", "剑气", "九天"]').
-- A generated tsvector column indexes these tokens directly using PostgreSQL's simple dictionary:
ALTER TABLE passages
ADD COLUMN IF NOT EXISTS fts_search_tokens tsvector
GENERATED ALWAYS AS (
    to_tsvector('simple', array_to_string(search_tokens, ' '))
) STORED;

CREATE INDEX IF NOT EXISTS idx_passages_cjk_fts_simple
ON passages
USING gin (fts_search_tokens);

-- Query matching using build_tsquery produced string:
-- SELECT id FROM passages WHERE fts_search_tokens @@ to_tsquery('simple', p_tsquery_string);


-- Option B: PGroonga Extension (When available on host platform)
-- CREATE EXTENSION IF NOT EXISTS pgroonga;
-- CREATE INDEX IF NOT EXISTS idx_passages_pgroonga
-- ON passages
-- USING pgroonga (text);
-- Query matching:
-- SELECT id FROM passages WHERE text &@~ '主角';
