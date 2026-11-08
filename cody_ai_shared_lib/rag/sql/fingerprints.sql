-- Reference SQL: Fingerprint Schema and Query (§4.7, §6.6, AC-RAG-06, AC-RAG-67)
-- StoreCapabilities:
--   supports_fingerprints=True
--   supports_revisions=True

-- Table schema example including fingerprint tracking columns
CREATE TABLE IF NOT EXISTS passages (
    id text PRIMARY KEY,
    doc_id text NOT NULL,
    section_id text,
    section_seq int NOT NULL DEFAULT 0,
    granularity text NOT NULL DEFAULT 'passage',
    char_start int NOT NULL DEFAULT 0,
    char_end int NOT NULL DEFAULT 0,
    text text,
    parent_chunk_id text,
    is_active boolean NOT NULL DEFAULT TRUE,

    -- Fingerprint & Ingestion metadata columns (§6.6, AC-RAG-06)
    content_hash text NOT NULL,
    embedding_model text NOT NULL,
    embedding_dim int NOT NULL,
    fts_signature text,
    token_estimate int,
    revision text DEFAULT 'v1',

    embedding vector(768),
    search_tokens text[],
    fts_search_tokens tsvector,
    created_at timestamptz DEFAULT NOW(),
    updated_at timestamptz DEFAULT NOW()
);

-- Fingerprint query for sync_document (§6.6, AC-RAG-06)
CREATE OR REPLACE FUNCTION get_document_fingerprints(p_doc_id text)
RETURNS TABLE (
    node_id text,
    content_hash text,
    embedding_signature text,
    fts_signature text
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id AS node_id,
        p.content_hash,
        (p.embedding_model || '|' || p.embedding_dim::text || '|RETRIEVAL_DOCUMENT|1.0.0') AS embedding_signature,
        p.fts_signature
    FROM passages p
    WHERE
        p.doc_id = p_doc_id
        AND p.is_active = TRUE AND {active_predicate};
END;
$$;

REVOKE EXECUTE ON FUNCTION get_document_fingerprints FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION get_document_fingerprints TO service_role;
