-- Reference SQL: filtered vector search (§4.2, §4.6 check 7, §4.7)
-- RAG_SPEC 1.2.0.
-- StoreCapabilities:
--   supports_vector_search=True
--   returns_raw_scores=True
--   supports_sequential_boundary=True
--
-- Contract:
--   * The owner predicate is mandatory: with the service role this RPC is the only isolation (§5.6).
--   * Ranked searches return passages only, never summaries (§4.2). Summaries come from
--     fetch_parent_summaries (sequential_boundary.sql).
--   * Boundary: a passage is a candidate when it starts before B; the engine cuts it at a sentence
--     end before B (§5.2). Per-document boundaries come as a jsonb map {doc_id: boundary}.
--   * Filtered recall (§4.6 check 7): pgvector >= 0.8 iterative index scans keep scanning until
--     match_count rows pass the filters, instead of returning fewer for one small document.
--   * Text by reference: chunk text is sliced with offsets relative to the chunk's own start.

CREATE OR REPLACE FUNCTION match_passages_vector(
    query_vector vector(768),
    p_tenant_or_user_id text,
    match_count int DEFAULT 8,
    p_domain_id text DEFAULT NULL,
    p_doc_ids text[] DEFAULT NULL,
    p_boundary int DEFAULT NULL,
    p_boundaries jsonb DEFAULT NULL,          -- {"doc_id": boundary, ...}
    p_min_similarity float DEFAULT 0.0
)
RETURNS TABLE (
    id text,
    doc_id text,
    section_id text,
    section_seq int,
    char_start int,
    char_end int,
    content text,
    similarity float
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
    IF p_tenant_or_user_id IS NULL OR p_tenant_or_user_id = '' THEN
        RAISE EXCEPTION 'match_passages_vector: owner is required' USING ERRCODE = '42501';
    END IF;
    PERFORM set_config('hnsw.iterative_scan', 'relaxed_order', true);   -- pgvector >= 0.8

    RETURN QUERY
    SELECT
        p.id,
        p.doc_id,
        p.section_id,
        p.section_seq,
        p.char_start,
        p.char_end,
        COALESCE(p.text, SUBSTRING(c.text FROM (p.char_start - c.char_start + 1) FOR (p.char_end - p.char_start))) AS content,
        (1.0 - (p.embedding <=> query_vector))::float AS similarity
    FROM passages p
    LEFT JOIN parent_chunks c ON c.id = p.parent_chunk_id
    WHERE
        p.tenant_or_user_id = p_tenant_or_user_id
        AND (p_domain_id IS NULL OR p.domain_id = p_domain_id)
        AND p.is_active = TRUE AND {active_predicate}
        AND p.is_summary = FALSE
        AND p.embedding IS NOT NULL
        AND (p_doc_ids IS NULL OR p.doc_id = ANY(p_doc_ids))
        AND (COALESCE((p_boundaries->>p.doc_id)::int, p_boundary) IS NULL
             OR p.char_start < COALESCE((p_boundaries->>p.doc_id)::int, p_boundary))
        AND (1.0 - (p.embedding <=> query_vector)) >= p_min_similarity
    ORDER BY p.embedding <=> query_vector ASC
    LIMIT match_count;
END;
$$;

REVOKE EXECUTE ON FUNCTION match_passages_vector FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION match_passages_vector TO service_role;
