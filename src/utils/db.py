"""Supabase/PostgreSQL connection helper."""

import psycopg2
import psycopg2.extras
import yaml
from pathlib import Path

def load_config():
    with open(Path(__file__).parent.parent.parent / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)

def get_connection():
    cfg = load_config()["vector_store"]
    return psycopg2.connect(
        host=cfg["host"], port=cfg["port"],
        dbname=cfg["database"], user=cfg["user"], password=cfg["password"]
    )

def execute_query(query, params=None, fetch=True):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(query, params)
    result = cur.fetchall() if fetch else None
    conn.commit()
    cur.close()
    conn.close()
    return result
