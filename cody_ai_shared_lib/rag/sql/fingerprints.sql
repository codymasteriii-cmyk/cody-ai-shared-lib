-- Reference SQL: table schema with owner and fingerprint columns, and the fingerprint query (§4.4.1, §4.7, §6.6)
-- RAG_SPEC 1.2.0. A template to copy: rename `passages`, `parent_chunks` and the columns to your schema,
-- then prove the result with the conformance kit (§4.6).
-- StoreCapabilities:
--   supports_fingerprints=True
--   supports_revisions=True
--   required_filter_keys={"tenant_or_user_id", ...}   -- every row has an owner (§6.4.2)

CREATE TABLE IF NOT EXISTS passages (
    id text PRIMARY KEY,                       -- node_id: an opaque key; store it, delete by it, never parse it
    doc_id text NOT NULL,
    tenant_or_user_id text NOT NULL,           -- the owner (§6.4.2); every read and write filters on it
    domain_id text,                            -- optional second partition (NewsIntelligence: user + domain)
    section_id text,
    section_seq int NOT NULL DEFAULT 0,
    granularity text NOT NULL DEFAULT 'passage',
    is_summary boolean NOT NULL DEFAULT FALSE, -- summaries come only from fetch_parent_summaries (§4.2)
    parent_kind text,                          -- 'section' | 'section_unit' for summaries
    unit_seq int,                              -- 0-based unit index for unit summaries; gaps allowed (§4.4.1)
    char_start int NOT NULL DEFAULT 0,         -- global document offsets
    char_end int NOT NULL DEFAULT 0,
    text text,                                 -- NULL for text by reference: sliced from parent_chunks
    parent_chunk_id text,
    is_active boolean NOT NULL DEFAULT TRUE,

    -- Fingerprint columns, persisted exactly as the engine sends them (§6.6). Never computed here.
    content_hash text NOT NULL,
    embedding_signature text,                  -- NULL: stored without a vector by design (embed_summaries=False)
    fts_signature text,
    token_estimate int,
    revision text,

    embedding vector(768),                     -- NULL exactly when embedding_signature is NULL
    search_tokens text[],
    fts_search_tokens tsvector,
    created_at timestamptz DEFAULT NOW(),
    updated_at timestamptz DEFAULT NOW()
);

-- Text by reference: the parent table holds the source text; char_start is the chunk's global offset
CREATE TABLE IF NOT EXISTS parent_chunks (
    id text PRIMARY KEY,
    doc_id text NOT NULL,
    tenant_or_user_id text NOT NULL,
    char_start int NOT NULL,
    text text NOT NULL
);

-- Fingerprints for sync_document and needs_reindex (§6.6): the stored values, as written
CREATE OR REPLACE FUNCTION get_document_fingerprints(p_tenant_or_user_id text, p_doc_id text)
RETURNS TABLE (
    node_id text,
    content_hash text,
    embedding_signature text,
    fts_signature text
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
    RETURN QUERY
    SELECT p.id, p.content_hash, p.embedding_signature, p.fts_signature
    FROM passages p
    WHERE p.tenant_or_user_id = p_tenant_or_user_id
      AND p.doc_id = p_doc_id
      AND p.is_active = TRUE AND {active_predicate};
END;
$$;

REVOKE EXECUTE ON FUNCTION get_document_fingerprints FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION get_document_fingerprints TO service_role;
