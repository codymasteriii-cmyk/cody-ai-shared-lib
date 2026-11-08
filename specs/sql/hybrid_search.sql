-- Reference SQL: Hybrid RRF Search with Reciprocal Rank Fusion (§4.7, AC-RAG-67)
-- StoreCapabilities:
--   supports_vector_search=True
--   supports_keyword_fts=True
--   supports_native_hybrid_rpc=True
--   returns_raw_scores=True
--   uses_analyzer_tokens=True
--   returns_snippets=True
--   supports_sequential_boundary=True

CREATE OR REPLACE FUNCTION match_passages_hybrid(
    query_vector vector(768),
    query_tsquery tsquery,
    match_count int DEFAULT 8,
    p_doc_id text DEFAULT NULL,
    p_boundary int DEFAULT NULL,
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
AS $$
BEGIN
    RETURN QUERY
    WITH
    -- Active row and boundary filter applied BEFORE per-branch limit
    -- Text-by-reference variant: content sliced from parent table parent_chunks
    -- (e.g. audiolib_passages over audiolib_article_chunks)
    vector_candidates AS (
        SELECT
            p.id,
            p.doc_id,
            p.section_id,
            p.section_seq,
            p.char_start,
            p.char_end,
            -- text by reference: SUBSTRING(c.text FROM (p.char_start + 1) FOR (p.char_end - p.char_start))
            COALESCE(p.text, SUBSTRING(c.text FROM (p.char_start + 1) FOR (p.char_end - p.char_start))) AS content,
            NULL::text AS snippet,
            (1.0 - (p.embedding <=> query_vector))::float AS vector_score,
            ROW_NUMBER() OVER (ORDER BY p.embedding <=> query_vector ASC) AS rank
        FROM passages p
        LEFT JOIN parent_chunks c ON c.id = p.parent_chunk_id
        WHERE
            -- Active row predicate
            p.is_active = TRUE AND {active_predicate}
            AND (p_doc_id IS NULL OR p.doc_id = p_doc_id)
            AND (p_boundary IS NULL OR p.char_start < p_boundary)
            AND (1.0 - (p.embedding <=> query_vector)) >= p_min_similarity
        ORDER BY p.embedding <=> query_vector ASC
        LIMIT match_count * 2
    ),
    fts_candidates AS (
        SELECT
            p.id,
            p.doc_id,
            p.section_id,
            p.section_seq,
            p.char_start,
            p.char_end,
            COALESCE(p.text, SUBSTRING(c.text FROM (p.char_start + 1) FOR (p.char_end - p.char_start))) AS content,
            -- Snippet expression placeholder
            {snippet_expr} AS snippet,
            ts_rank_cd(p.fts_search_tokens, query_tsquery)::float AS keyword_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(p.fts_search_tokens, query_tsquery) DESC) AS rank
        FROM passages p
        LEFT JOIN parent_chunks c ON c.id = p.parent_chunk_id
        WHERE
            p.is_active = TRUE AND {active_predicate}
            AND (p_doc_id IS NULL OR p.doc_id = p_doc_id)
            AND (p_boundary IS NULL OR p.char_start < p_boundary)
            AND p.fts_search_tokens @@ query_tsquery
        ORDER BY ts_rank_cd(p.fts_search_tokens, query_tsquery) DESC
        LIMIT match_count * 2
    ),
    fused AS (
        SELECT
            COALESCE(v.id, f.id) AS id,
            COALESCE(v.doc_id, f.doc_id) AS doc_id,
            COALESCE(v.section_id, f.section_id) AS section_id,
            COALESCE(v.section_seq, f.section_seq) AS section_seq,
            COALESCE(v.char_start, f.char_start) AS char_start,
            COALESCE(v.char_end, f.char_end) AS char_end,
            COALESCE(v.content, f.content) AS content,
            COALESCE(f.snippet, v.snippet) AS snippet,
            v.vector_score,
            f.keyword_score,
            v.rank::int AS vector_rank,
            f.rank::int AS keyword_rank,
            (
                COALESCE(p_w_vec / (p_rrf_k + v.rank), 0.0) +
                COALESCE(p_w_fts / (p_rrf_k + f.rank), 0.0)
            )::float AS rrf_score
        FROM vector_candidates v
        FULL OUTER JOIN fts_candidates f ON v.id = f.id
    )
    SELECT
        fused.id,
        fused.doc_id,
        fused.section_id,
        fused.section_seq,
        fused.char_start,
        fused.char_end,
        fused.content,
        fused.snippet,
        fused.vector_score,
        fused.keyword_score,
        fused.vector_rank,
        fused.keyword_rank,
        fused.rrf_score
    FROM fused
    ORDER BY fused.rrf_score DESC
    LIMIT match_count;
END;
$$;

-- Security / Execution Rights (§4.7)
REVOKE EXECUTE ON FUNCTION match_passages_hybrid FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION match_passages_hybrid TO service_role;
