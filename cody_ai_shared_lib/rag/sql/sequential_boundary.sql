-- Reference SQL: Sequential Boundary Enforcement (§4.7, §5.2, AC-RAG-24, AC-RAG-25, AC-RAG-67)
-- StoreCapabilities:
--   supports_sequential_boundary=True
--   supports_parent_summaries=True

-- Type for per-document sequential boundary array (§3.6, AC-RAG-25)
CREATE TYPE doc_boundary_spec AS (
    doc_id text,
    boundary int
);

CREATE OR REPLACE FUNCTION match_with_sequential_boundary(
    p_boundaries doc_boundary_spec[] DEFAULT NULL,
    p_single_doc_id text DEFAULT NULL,
    p_single_boundary int DEFAULT NULL
)
RETURNS TABLE (
    id text,
    doc_id text,
    granularity text,
    char_start int,
    char_end int,
    content text
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id,
        p.doc_id,
        p.granularity,
        p.char_start,
        p.char_end,
        -- text by reference
        COALESCE(p.text, SUBSTRING(c.text FROM (p.char_start + 1) FOR (p.char_end - p.char_start))) AS content
    FROM passages p
    LEFT JOIN parent_chunks c ON c.id = p.parent_chunk_id
    LEFT JOIN unnest(p_boundaries) b ON b.doc_id = p.doc_id
    WHERE
        p.is_active = TRUE AND {active_predicate}
        -- Single doc boundary shorthand
        AND (
            p_single_boundary IS NULL
            OR p_single_doc_id IS NULL
            OR p.doc_id != p_single_doc_id
            OR (
                -- Summaries admitted only if char_end <= boundary (§5.2)
                CASE WHEN p.granularity = 'parent_summary' THEN p.char_end <= p_single_boundary
                ELSE p.char_start < p_single_boundary END
            )
        )
        -- Per-document boundary map (AC-RAG-25)
        AND (
            b.boundary IS NULL
            OR (
                CASE WHEN p.granularity = 'parent_summary' THEN p.char_end <= b.boundary
                ELSE p.char_start < b.boundary END
            )
        );
END;
$$;

-- Security / Execution Rights (§4.7)
REVOKE EXECUTE ON FUNCTION match_with_sequential_boundary FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION match_with_sequential_boundary TO service_role;
