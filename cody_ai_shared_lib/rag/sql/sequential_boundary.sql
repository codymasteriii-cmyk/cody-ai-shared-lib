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

-- Reference SQL: summaries under the sequential boundary (§4.2, §5.2, §4.7)
-- RAG_SPEC 1.2.0. Backs fetch_parent_summaries. Passage searches apply the boundary themselves
-- (vector_search.sql, hybrid_search.sql); this file covers the summary layer.
-- StoreCapabilities:
--   supports_sequential_boundary=True
--   supports_parent_summaries=True
--   supports_summary_levels=True       -- when p_level is honoured
--
-- The spoiler rule for summaries (§5.2 rule 3): a summary is admitted only when its WHOLE span ends
-- before the boundary (char_end <= B). Admitting by start offset would let the summary of the
-- section the reader is in describe text they have not reached: the 1.0.0 leak (C-01).
-- Per-document boundaries come as a jsonb map {doc_id: boundary}; a single boundary applies to the rest.

CREATE OR REPLACE FUNCTION fetch_parent_summaries(
    p_tenant_or_user_id text,
    p_doc_ids text[] DEFAULT NULL,
    p_domain_id text DEFAULT NULL,
    p_boundary int DEFAULT NULL,
    p_boundaries jsonb DEFAULT NULL,
    p_level int DEFAULT NULL
)
RETURNS TABLE (
    id text,
    doc_id text,
    section_id text,
    section_seq int,
    parent_kind text,
    unit_seq int,
    char_start int,
    char_end int,
    content text
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions, pg_temp
AS $$
BEGIN
    IF p_tenant_or_user_id IS NULL OR p_tenant_or_user_id = '' THEN
        RAISE EXCEPTION 'fetch_parent_summaries: owner is required' USING ERRCODE = '42501';
    END IF;

    RETURN QUERY
    SELECT p.id, p.doc_id, p.section_id, p.section_seq, p.parent_kind, p.unit_seq,
           p.char_start, p.char_end, p.text AS content
    FROM passages p
    WHERE
        p.tenant_or_user_id = p_tenant_or_user_id
        AND (p_domain_id IS NULL OR p.domain_id = p_domain_id)
        AND p.is_active = TRUE AND {active_predicate}
        AND p.is_summary = TRUE
        AND (p_doc_ids IS NULL OR p.doc_id = ANY(p_doc_ids))
        AND (p_level IS NULL
             OR (p_level = 1 AND p.parent_kind = 'section')
             OR (p_level = 0 AND p.parent_kind = 'section_unit'))
        AND (COALESCE((p_boundaries->>p.doc_id)::int, p_boundary) IS NULL
             OR p.char_end <= COALESCE((p_boundaries->>p.doc_id)::int, p_boundary))
    ORDER BY p.doc_id, p.char_start, p.section_seq, p.unit_seq NULLS FIRST;
END;
$$;

REVOKE EXECUTE ON FUNCTION fetch_parent_summaries FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION fetch_parent_summaries TO service_role;
