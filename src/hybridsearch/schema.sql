CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS chunks;

CREATE TABLE chunks (
    chunk_id      text PRIMARY KEY,
    doc_id        text NOT NULL,
    plan          text,
    heading_path  text NOT NULL DEFAULT '',
    content       text NOT NULL,
    embedding     vector(384) NOT NULL,
    tsv           tsvector GENERATED ALWAYS AS (
                      setweight(to_tsvector('english', coalesce(heading_path, '')), 'A') ||
                      setweight(to_tsvector('english', content), 'B')
                  ) STORED
);

CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);

CREATE INDEX chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX chunks_plan_idx ON chunks (plan);
