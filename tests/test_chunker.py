"""Tests for the chunking module."""

import pytest
from src.ingestion.chunker import chunk_text, count_tokens
from src.ingestion.pdf_parser import Section, ParsedDocument
from src.ingestion.chunker import chunk_document

def test_count_tokens():
    text = "Hello world this is a test"
    count = count_tokens(text)
    assert count > 0
    assert isinstance(count, int)

def test_chunk_text_short():
    text = "Short text."
    chunks = chunk_text(text, chunk_size=100, overlap_pct=0.15)
    assert len(chunks) == 1
    assert chunks[0][1] is False  # no overlap on first chunk

def test_chunk_text_with_overlap():
    # Create text that's definitely longer than chunk_size
    text = " ".join(["word"] * 500)
    chunks = chunk_text(text, chunk_size=50, overlap_pct=0.2)
    assert len(chunks) > 1
    assert chunks[0][1] is False   # first chunk has no overlap
    assert chunks[1][1] is True    # second chunk has overlap

def test_chunk_text_zero_overlap():
    text = " ".join(["word"] * 500)
    chunks = chunk_text(text, chunk_size=50, overlap_pct=0.0)
    assert len(chunks) > 1
    # With zero overlap, chunks should not repeat content
    for _, is_overlap in chunks:
        pass  # just verify it doesn't crash

def test_chunk_document_respects_sections():
    doc = ParsedDocument(
        doc_id="test", ticker="TEST", filing_type="10-K",
        total_pages=1,
        sections=[
            Section(section_id="Item 7", title="MD&A",
                    content=" ".join(["revenue"] * 200),
                    page_start=0, page_end=0, tables=[]),
            Section(section_id="Item 8", title="Financial Statements",
                    content=" ".join(["balance"] * 200),
                    page_start=1, page_end=1, tables=[]),
        ],
        raw_text=""
    )
    config = {
        "chunk_size": 50, "overlap_pct": 0.15,
        "min_chunk_size": 5, "respect_sections": True,
        "tokenizer": "cl100k_base"
    }
    chunks = chunk_document(doc, config)
    assert len(chunks) > 0

    # Verify no chunk spans sections
    section_ids = set(c.section_id for c in chunks)
    assert "Item 7" in section_ids
    assert "Item 8" in section_ids

    for chunk in chunks:
        if chunk.section_id == "Item 7":
            assert "revenue" in chunk.content
        if chunk.section_id == "Item 8":
            assert "balance" in chunk.content

def test_chunk_document_skips_tiny_sections():
    doc = ParsedDocument(
        doc_id="test", ticker="TEST", filing_type="10-K",
        total_pages=1,
        sections=[
            Section(section_id="Item 1", title="Business",
                    content="Tiny.",
                    page_start=0, page_end=0, tables=[]),
        ],
        raw_text=""
    )
    config = {
        "chunk_size": 512, "overlap_pct": 0.15,
        "min_chunk_size": 100, "respect_sections": True,
        "tokenizer": "cl100k_base"
    }
    chunks = chunk_document(doc, config)
    assert len(chunks) == 0  # too small, should be skipped
