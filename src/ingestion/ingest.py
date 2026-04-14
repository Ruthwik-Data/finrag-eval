"""Full ingestion pipeline: parse → chunk → embed → store."""

import json
import click
from pathlib import Path
from rich.console import Console
from rich.progress import track

from src.ingestion.pdf_parser import parse_pdf, parse_html_filing
from src.ingestion.chunker import chunk_document
from src.utils.ollama_client import OllamaEmbedder
from src.utils.db import get_connection

console = Console()

def store_chunks(chunks, embeddings):
    conn = get_connection()
    cur = conn.cursor()

    for chunk, embedding in zip(chunks, embeddings):
        cur.execute("""
            INSERT INTO document_chunks
            (doc_id, ticker, filing_type, section_id, section_title,
             chunk_index, content, token_count, overlap_prev, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            chunk.doc_id, chunk.ticker, chunk.filing_type,
            chunk.section_id, chunk.section_title,
            chunk.chunk_index, chunk.content, chunk.token_count,
            chunk.overlap_prev, json.dumps(chunk.metadata),
            str(embedding)
        ))

    conn.commit()
    cur.close()
    conn.close()

@click.command()
@click.option("--input", "input_path", required=True, help="Path to PDF or HTML filing")
@click.option("--ticker", default="", help="Stock ticker")
@click.option("--filing-type", default="10-K", help="Filing type")
def ingest(input_path, ticker, filing_type):
    path = Path(input_path)
    console.print(f"\n[bold blue]Ingesting:[/bold blue] {path.name}")

    # 1. Parse
    console.print("[dim]Step 1/4: Parsing document...[/dim]")
    if path.suffix.lower() in (".htm", ".html"):
        doc = parse_html_filing(str(path), ticker=ticker, filing_type=filing_type)
    else:
        doc = parse_pdf(str(path), ticker=ticker, filing_type=filing_type)
    console.print(f"  Sections found: {len(doc.sections)}")
    for s in doc.sections[:5]:
        console.print(f"    {s.section_id}: {s.title[:60]} ({len(s.content)} chars)")
    if len(doc.sections) > 5:
        console.print(f"    ... and {len(doc.sections) - 5} more")

    # 2. Chunk
    console.print("[dim]Step 2/4: Chunking...[/dim]")
    chunks = chunk_document(doc)
    console.print(f"  Chunks created: {len(chunks)}")
    overlap_count = sum(1 for c in chunks if c.overlap_prev)
    console.print(f"  Overlapping chunks: {overlap_count} ({overlap_count/len(chunks)*100:.0f}%)")
    avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
    console.print(f"  Avg tokens/chunk: {avg_tokens:.0f}")

    # 3. Embed
    console.print("[dim]Step 3/4: Embedding with Ollama...[/dim]")
    embedder = OllamaEmbedder()
    texts = [c.content for c in chunks]
    embeddings = []
    batch_size = 32
    for i in track(range(0, len(texts), batch_size), description="  Embedding..."):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedder.embed_batch(batch)
        embeddings.extend(batch_embeddings)
    console.print(f"  Embeddings generated: {len(embeddings)} (dim={len(embeddings[0])})")

    # 4. Store
    console.print("[dim]Step 4/4: Storing in Supabase pgvector...[/dim]")
    store_chunks(chunks, embeddings)
    console.print(f"  Stored {len(chunks)} chunks in document_chunks table")

    console.print(f"\n[bold green]Done![/bold green] {path.name} → {len(chunks)} chunks ingested.\n")

if __name__ == "__main__":
    ingest()
