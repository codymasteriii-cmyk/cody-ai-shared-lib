-- Reference SQL: Atomic Upsert Section RPC (§4.7, §4.4, RAG-ING-10, AC-RAG-66, AC-RAG-67)
-- StoreCapabilities:
--   supports_parent_summaries=True
--   supports_fingerprints=True

CREATE OR REPLACE FUNCTION upsert_section_nodes(
    p_doc_id text,
    p_section_seq int,
    p_nodes jsonb,
    p_summary jsonb DEFAULT NULL
)
RETURNS int
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    written_count int := 0;
BEGIN
    -- Atomic deletion of existing section rows to guarantee consistency (RAG-ING-10)
    -- Applies active-row check if soft deletes used: AND {active_predicate}
    DELETE FROM passages
    WHERE doc_id = p_doc_id AND section_seq = p_section_seq;

    -- Insert new passage nodes
    INSERT INTO passages (
        id,
        doc_id,
        section_id,
        section_seq,
        granularity,
        char_start,
        char_end,
        text,
        content_hash,
        embedding_model,
        embedding_dim,
        fts_signature,
        token_estimate,
        embedding,
        search_tokens,
        fts_search_tokens,
        is_active
    )
    SELECT
        item->>'id',
        p_doc_id,
        item->>'section_id',
        p_section_seq,
        COALESCE(item->>'granularity', 'passage'),
        (item->>'char_start')::int,
        (item->>'char_end')::int,
        item->>'text',
        item->>'content_hash',
        item->>'embedding_model',
        (item->>'embedding_dim')::int,
        item->>'fts_signature',
        (item->>'token_estimate')::int,
        (item->>'embedding')::vector(768),
        ARRAY(SELECT jsonb_array_elements_text(item->'search_tokens')),
        to_tsvector('simple', COALESCE(item->>'search_tokens_text', '')),
        TRUE
    FROM jsonb_array_elements(p_nodes) AS item;

    GET DIAGNOSTICS written_count = ROW_COUNT;

    -- Insert section summary if provided
    IF p_summary IS NOT NULL THEN
        INSERT INTO passages (
            id,
            doc_id,
            section_id,
            section_seq,
            granularity,
            char_start,
            char_end,
            text,
            content_hash,
            embedding_model,
            embedding_dim,
            fts_signature,
            token_estimate,
            embedding
        )
        VALUES (
            p_summary->>'id',
            p_doc_id,
            p_summary->>'section_id',
            p_section_seq,
            'parent_summary',
            (p_summary->>'char_start')::int,
            (p_summary->>'char_end')::int,
            p_summary->>'text',
            p_summary->>'content_hash',
            p_summary->>'embedding_model',
            (p_summary->>'embedding_dim')::int,
            p_summary->>'fts_signature',
            (p_summary->>'token_estimate')::int,
            (p_summary->>'embedding')::vector(768)
        );
        written_count := written_count + 1;
    END IF;

    RETURN written_count;
END;
$$;

REVOKE EXECUTE ON FUNCTION upsert_section_nodes FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION upsert_section_nodes TO service_role;
