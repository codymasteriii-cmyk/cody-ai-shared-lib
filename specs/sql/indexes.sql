-- Reference SQL: Index Definitions (§4.7, AC-RAG-67)
-- StoreCapabilities:
--   supports_vector_search=True
--   supports_keyword_fts=True
--   supports_sequential_boundary=True

-- 1. HNSW Index on embedding vector column for approximate nearest neighbours
CREATE INDEX IF NOT EXISTS idx_passages_embedding_hnsw
ON passages
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 2. GIN Index on FTS search tokens tsvector
CREATE INDEX IF NOT EXISTS idx_passages_fts_tokens_gin
ON passages
USING gin (fts_search_tokens);

-- 3. B-Tree on (doc_id, char_start) for exact sequential boundary and range filtering
CREATE INDEX IF NOT EXISTS idx_passages_doc_char_start
ON passages (doc_id, char_start);

-- 4. B-Tree on tenant or domain scope column
CREATE INDEX IF NOT EXISTS idx_passages_doc_active
ON passages (doc_id, is_active);
