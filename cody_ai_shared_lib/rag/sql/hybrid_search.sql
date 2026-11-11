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

-- Reference SQL: hybrid search with Reciprocal Rank Fusion in one round trip (§4.2, §4.7, §9.1)
-- RAG_SPEC 1.2.0.
-- StoreCapabilities:
--   supports_vector_search=True
--   supports_keyword_fts=True
--   supports_native_hybrid_rpc=True
--   returns_raw_scores=True
--   uses_analyzer_tokens=True          -- the adapter builds query_tsquery with rag.text.build_tsquery
--   returns_snippets=True
--   supports_sequential_boundary=True
--
-- Contract (same rules as vector_search.sql):
--   * Mandatory owner predicate; passages only (never summaries); a passage is a candidate when it
--     starts before its document's boundary; chunk text sliced with chunk-relative offsets.
--   * Every filter applies in BOTH branches BEFORE each branch's limit (match_count * 2).
--   * p_min_similarity applies to the vector branch only, so a keyword-only hit survives (§9.1).
--   * Raw vector_score / keyword_score and both ranks are returned alongside the RRF score.

CREATE OR REPLACE FUNCTION match_passages_hybrid(
    query_vector vector(768),
    query_tsquery tsquery,
    p_tenant_or_user_id text,
    match_count int DEFAULT 8,
    p_domain_id text DEFAULT NULL,
    p_doc_ids text[] DEFAULT NULL,
    p_boundary int DEFAULT NULL,
    p_boundaries jsonb DEFAULT NULL,          -- {"doc_id": boundary, ...}
    p_min_similarity float DEFAULT 0.60,
    p_rrf_k int DEFAULT 60,
    p_w_vec float DEFAULT 1.0,
    p_w_fts float DEFAULT 1.0
)
RETURNS TABLE (
    id text,
    doc_id text,
    section_id text,
    section_seq int,
    char_start int,
    char_end int,
    content text,
    snippet text,
    vector_score float,
    keyword_score float,
    vector_rank int,
    keyword_rank int,
    rrf_score float
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions, pg_temp
AS $$
BEGIN
    IF p_tenant_or_user_id IS NULL OR p_tenant_or_user_id = '' THEN
        RAISE EXCEPTION 'match_passages_hybrid: owner is required' USING ERRCODE = '42501';
    END IF;
    PERFORM set_config('hnsw.iterative_scan', 'relaxed_order', true);   -- pgvector >= 0.8

    RETURN QUERY
    WITH
    scoped AS (
        -- Every hard filter, once, before either branch ranks or limits anything
        SELECT p.*
        FROM passages p
        WHERE
            p.tenant_or_user_id = p_tenant_or_user_id
            AND (p_domain_id IS NULL OR p.domain_id = p_domain_id)
            AND p.is_active = TRUE AND {active_predicate}
            AND p.is_summary = FALSE
            AND (p_doc_ids IS NULL OR p.doc_id = ANY(p_doc_ids))
            AND (COALESCE((p_boundaries->>p.doc_id)::int, p_boundary) IS NULL
                 OR p.char_start < COALESCE((p_boundaries->>p.doc_id)::int, p_boundary))
    ),
    vector_branch AS (
        SELECT s.id,
               (1.0 - (s.embedding <=> query_vector))::float AS v_score,
               ROW_NUMBER() OVER (ORDER BY s.embedding <=> query_vector ASC)::int AS v_rank
        FROM scoped s
        WHERE s.embedding IS NOT NULL
          AND (1.0 - (s.embedding <=> query_vector)) >= p_min_similarity
        ORDER BY s.embedding <=> query_vector ASC
        LIMIT match_count * 2
    ),
    keyword_branch AS (
        SELECT s.id,
               ts_rank_cd(s.fts_search_tokens, query_tsquery)::float AS k_score,
               ROW_NUMBER() OVER (ORDER BY ts_rank_cd(s.fts_search_tokens, query_tsquery) DESC, s.id)::int AS k_rank
        FROM scoped s
        WHERE s.fts_search_tokens @@ query_tsquery
        ORDER BY ts_rank_cd(s.fts_search_tokens, query_tsquery) DESC, s.id
        LIMIT match_count * 2
    ),
    fused AS (
        SELECT COALESCE(v.id, k.id) AS id,
               v.v_score, k.k_score, v.v_rank, k.k_rank,
               (COALESCE(p_w_vec / (p_rrf_k + v.v_rank), 0.0)
                + COALESCE(p_w_fts / (p_rrf_k + k.k_rank), 0.0))::float AS rrf
        FROM vector_branch v
        FULL OUTER JOIN keyword_branch k ON k.id = v.id
    )
    SELECT
        p.id,
        p.doc_id,
        p.section_id,
        p.section_seq,
        p.char_start,
        p.char_end,
        COALESCE(p.text, SUBSTRING(c.text FROM (p.char_start - c.char_start + 1) FOR (p.char_end - p.char_start))) AS content,
        CASE WHEN f.k_rank IS NULL THEN NULL
             ELSE ts_headline('simple',
                    COALESCE(p.text, SUBSTRING(c.text FROM (p.char_start - c.char_start + 1) FOR (p.char_end - p.char_start))),
                    query_tsquery, 'MaxFragments=1, MaxWords=30, MinWords=10')
        END AS snippet,
        f.v_score,
        f.k_score,
        f.v_rank,
        f.k_rank,
        f.rrf
    FROM fused f
    JOIN passages p ON p.id = f.id
    LEFT JOIN parent_chunks c ON c.id = p.parent_chunk_id
    ORDER BY f.rrf DESC, p.id
    LIMIT match_count;
END;
$$;

REVOKE EXECUTE ON FUNCTION match_passages_hybrid FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION match_passages_hybrid TO service_role;
