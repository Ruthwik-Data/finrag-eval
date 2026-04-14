"""Vector store operations for Supabase pgvector."""

import json
from src.utils.db import get_connection
from src.utils.ollama_client import OllamaEmbedder

class VectorStore:
    def __init__(self):
        self.embedder = OllamaEmbedder()

    def search(self, query: str, top_k: int = 8, ticker: str = None) -> list[dict]:
        query_embedding = self.embedder.embed(query)
        conn = get_connection()
        cur = conn.cursor()

        if ticker:
            cur.execute(
                "SELECT * FROM match_chunks(%s::vector, %s, %s)",
                (str(query_embedding), top_k, ticker)
            )
        else:
            cur.execute(
                "SELECT * FROM match_chunks(%s::vector, %s)",
                (str(query_embedding), top_k)
            )

        columns = [desc[0] for desc in cur.description]
        results = [dict(zip(columns, row)) for row in cur.fetchall()]

        cur.close()
        conn.close()
        return results

    def get_chunk_count(self, ticker: str = None) -> int:
        conn = get_connection()
        cur = conn.cursor()
        if ticker:
            cur.execute("SELECT COUNT(*) FROM document_chunks WHERE ticker = %s", (ticker,))
        else:
            cur.execute("SELECT COUNT(*) FROM document_chunks")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count

    def get_sections(self, doc_id: str) -> list[dict]:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT section_id, section_title,
                   COUNT(*) as chunk_count,
                   SUM(token_count) as total_tokens
            FROM document_chunks
            WHERE doc_id = %s
            GROUP BY section_id, section_title
            ORDER BY MIN(chunk_index)
        """, (doc_id,))
        columns = [desc[0] for desc in cur.description]
        results = [dict(zip(columns, row)) for row in cur.fetchall()]
        cur.close()
        conn.close()
        return results
