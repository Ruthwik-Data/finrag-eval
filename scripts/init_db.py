"""Initialize Supabase local database with pgvector extension and tables."""

import psycopg2
import yaml
from pathlib import Path

def load_config():
    with open(Path(__file__).parent.parent / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)

def init_db():
    cfg = load_config()["vector_store"]
    conn = psycopg2.connect(
        host=cfg["host"], port=cfg["port"],
        dbname=cfg["database"], user=cfg["user"], password=cfg["password"]
    )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            doc_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            filing_type TEXT NOT NULL,
            section_id TEXT,
            section_title TEXT,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER,
            overlap_prev BOOLEAN DEFAULT FALSE,
            metadata JSONB DEFAULT '{}',
            embedding vector(768),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
        ON document_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 50);
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(doc_id);
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_ticker ON document_chunks(ticker);
    """)

    # pgvector match function for retrieval
    cur.execute("""
        CREATE OR REPLACE FUNCTION match_chunks(
            query_embedding vector(768),
            match_count INT DEFAULT 8,
            filter_ticker TEXT DEFAULT NULL
        )
        RETURNS TABLE(
            id INT,
            doc_id TEXT,
            ticker TEXT,
            section_id TEXT,
            section_title TEXT,
            content TEXT,
            metadata JSONB,
            similarity FLOAT
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                dc.id, dc.doc_id, dc.ticker, dc.section_id,
                dc.section_title, dc.content, dc.metadata,
                1 - (dc.embedding <=> query_embedding) AS similarity
            FROM document_chunks dc
            WHERE (filter_ticker IS NULL OR dc.ticker = filter_ticker)
            ORDER BY dc.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
    """)

    # Eval results table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id SERIAL PRIMARY KEY,
            run_id TEXT NOT NULL,
            config JSONB NOT NULL,
            metrics JSONB NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS eval_results (
            id SERIAL PRIMARY KEY,
            run_id TEXT NOT NULL,
            question TEXT NOT NULL,
            expected_answer TEXT,
            actual_answer TEXT,
            retrieval_context JSONB,
            scores JSONB NOT NULL,
            dedup_scores JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cur.close()
    conn.close()
    print("Database initialized: pgvector extension, tables, indexes, match function.")

if __name__ == "__main__":
    init_db()
