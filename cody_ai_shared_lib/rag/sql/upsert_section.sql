-- Reference SQL: one section write, in one transaction (§4.4.1, §4.7, RAG-ING-10)
-- RAG_SPEC 1.2.0. Backs an adapter's `upsert_section(write: SectionWrite)` override.
-- StoreCapabilities:
--   supports_parent_summaries=True
--   supports_fingerprints=True
--
-- Contract (§4.4.1):
--   * Idempotent on id (node_id). NEVER deletes: during sync a write holds only the changed rows of a
--     section, and every other row must stay. Deletion is a separate delete_nodes RPC.
--   * A reader sees none or all of the write (one function body = one transaction).
--   * Fingerprints are stored as given; the store computes nothing.
--   * A document belongs to one owner. A write into a document another owner holds is refused (§5.6).
--
-- The adapter serializes SectionWrite into three jsonb arrays of flat items:
--   {"id", "section_id", "granularity", "char_start", "char_end", "text", "parent_chunk_id",
--    "content_hash", "embedding_signature", "fts_signature", "token_estimate", "revision",
--    "embedding": [..] | null, "search_tokens": [..], "unit_seq": n | null}
-- and the owner as {"tenant_or_user_id": "...", "domain_id": "..." | null}.

CREATE OR REPLACE FUNCTION upsert_section(
    p_doc_id text,
    p_section_seq int,
    p_owner jsonb,
    p_passages jsonb DEFAULT '[]'::jsonb,
    p_unit_summaries jsonb DEFAULT '[]'::jsonb,
    p_section_summary jsonb DEFAULT NULL
)
RETURNS int
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_owner text := p_owner->>'tenant_or_user_id';
    v_existing_owner text;
    v_expected int;
    v_written int := 0;
BEGIN
    IF v_owner IS NULL OR v_owner = '' THEN
        RAISE EXCEPTION 'upsert_section: owner is required' USING ERRCODE = '42501';
    END IF;

    -- A document never changes owner: refuse a write into another owner's document (§5.6)
    SELECT p.tenant_or_user_id INTO v_existing_owner
    FROM passages p WHERE p.doc_id = p_doc_id LIMIT 1;
    IF v_existing_owner IS NOT NULL AND v_existing_owner <> v_owner THEN
        RAISE EXCEPTION 'upsert_section: document % belongs to another owner', p_doc_id USING ERRCODE = '42501';
    END IF;

    v_expected := jsonb_array_length(COALESCE(p_passages, '[]'::jsonb))
                + jsonb_array_length(COALESCE(p_unit_summaries, '[]'::jsonb))
                + CASE WHEN p_section_summary IS NULL THEN 0 ELSE 1 END;

    WITH items AS (
        SELECT item, 'passage'::text AS kind FROM jsonb_array_elements(COALESCE(p_passages, '[]'::jsonb)) AS item
        UNION ALL
        SELECT item, 'section_unit' FROM jsonb_array_elements(COALESCE(p_unit_summaries, '[]'::jsonb)) AS item
        UNION ALL
        SELECT p_section_summary, 'section' WHERE p_section_summary IS NOT NULL
    )
    INSERT INTO passages AS t (
        id, doc_id, tenant_or_user_id, domain_id, section_id, section_seq, granularity,
        is_summary, parent_kind, unit_seq, char_start, char_end, text, parent_chunk_id, is_active,
        content_hash, embedding_signature, fts_signature, token_estimate, revision,
        embedding, search_tokens, fts_search_tokens, updated_at
    )
    SELECT
        i.item->>'id',
        p_doc_id,
        v_owner,
        p_owner->>'domain_id',
        i.item->>'section_id',
        p_section_seq,
        CASE WHEN i.kind = 'passage' THEN COALESCE(i.item->>'granularity', 'passage') ELSE 'parent_summary' END,
        i.kind <> 'passage',
        CASE WHEN i.kind = 'passage' THEN NULL ELSE i.kind END,
        (i.item->>'unit_seq')::int,
        (i.item->>'char_start')::int,
        (i.item->>'char_end')::int,
        i.item->>'text',
        i.item->>'parent_chunk_id',
        TRUE,
        i.item->>'content_hash',
        i.item->>'embedding_signature',
        i.item->>'fts_signature',
        (i.item->>'token_estimate')::int,
        i.item->>'revision',
        CASE WHEN i.item->'embedding' IS NULL OR jsonb_typeof(i.item->'embedding') = 'null'
             THEN NULL ELSE (i.item->>'embedding')::vector(768) END,
        ARRAY(SELECT jsonb_array_elements_text(COALESCE(i.item->'search_tokens', '[]'::jsonb))),
        to_tsvector('simple', array_to_string(
            ARRAY(SELECT jsonb_array_elements_text(COALESCE(i.item->'search_tokens', '[]'::jsonb))), ' ')),
        NOW()
    FROM items i
    ON CONFLICT (id) DO UPDATE SET
        section_id = EXCLUDED.section_id,
        granularity = EXCLUDED.granularity,
        is_summary = EXCLUDED.is_summary,
        parent_kind = EXCLUDED.parent_kind,
        unit_seq = EXCLUDED.unit_seq,
        char_start = EXCLUDED.char_start,
        char_end = EXCLUDED.char_end,
        text = EXCLUDED.text,
        parent_chunk_id = EXCLUDED.parent_chunk_id,
        is_active = TRUE,
        content_hash = EXCLUDED.content_hash,
        embedding_signature = EXCLUDED.embedding_signature,
        fts_signature = EXCLUDED.fts_signature,
        token_estimate = EXCLUDED.token_estimate,
        revision = EXCLUDED.revision,
        embedding = EXCLUDED.embedding,
        search_tokens = EXCLUDED.search_tokens,
        fts_search_tokens = EXCLUDED.fts_search_tokens,
        updated_at = NOW()
    WHERE t.tenant_or_user_id = EXCLUDED.tenant_or_user_id;   -- an id held by another owner is never updated

    GET DIAGNOSTICS v_written = ROW_COUNT;
    IF v_written <> v_expected THEN
        -- Some id belongs to another owner: abort the whole write
        RAISE EXCEPTION 'upsert_section: % of % rows belong to another owner', v_expected - v_written, v_expected
            USING ERRCODE = '42501';
    END IF;
    RETURN v_written;
END;
$$;

REVOKE EXECUTE ON FUNCTION upsert_section FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION upsert_section TO service_role;

-- Deletion is its own RPC, scoped to the owner (sync deletes vanished nodes last, §6.6)
CREATE OR REPLACE FUNCTION delete_nodes(p_tenant_or_user_id text, p_node_ids text[])
RETURNS int
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_deleted int;
BEGIN
    DELETE FROM passages p
    WHERE p.tenant_or_user_id = p_tenant_or_user_id AND p.id = ANY(p_node_ids);
    GET DIAGNOSTICS v_deleted = ROW_COUNT;
    RETURN v_deleted;
END;
$$;

REVOKE EXECUTE ON FUNCTION delete_nodes FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION delete_nodes TO service_role;
