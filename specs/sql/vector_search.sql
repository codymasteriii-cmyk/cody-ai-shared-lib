-- Reference SQL: Filtered Vector Search (§4.7, AC-RAG-67)
-- StoreCapabilities:
--   supports_vector_search=True
--   returns_raw_scores=True
--   supports_sequential_boundary=True

CREATE OR REPLACE FUNCTION match_passages_vector(
    query_vector vector(768),
    match_count int DEFAULT 8,
    p_doc_id text DEFAULT NULL,
    p_boundary int DEFAULT NULL,
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
AS $$
BEGIN
    -- Avoid filtered-recall trap (§4.6 #7):
    -- If p_doc_id is provided, exact table scan / indexed btree on (doc_id, embedding)
    -- is preferred over early-terminated approximate HNSW scan.
    RETURN QUERY
    SELECT
        p.id,
        p.doc_id,
        p.section_id,
        p.section_seq,
        p.char_start,
        p.char_end,
        -- text by reference variant
        COALESCE(p.text, SUBSTRING(c.text FROM (p.char_start + 1) FOR (p.char_end - p.char_start))) AS content,
        (1.0 - (p.embedding <=> query_vector))::float AS similarity
    FROM passages p
    LEFT JOIN parent_chunks c ON c.id = p.parent_chunk_id
    WHERE
        p.is_active = TRUE AND {active_predicate}
        AND (p_doc_id IS NULL OR p.doc_id = p_doc_id)
        AND (p_boundary IS NULL OR p.char_start < p_boundary)
        AND (1.0 - (p.embedding <=> query_vector)) >= p_min_similarity
    ORDER BY p.embedding <=> query_vector ASC
    LIMIT match_count;
END;
$$;

-- Security / Execution Rights (§4.7)
REVOKE EXECUTE ON FUNCTION match_passages_vector FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION match_passages_vector TO service_role;
