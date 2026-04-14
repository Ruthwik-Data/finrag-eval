"""Section-aware text chunking with configurable overlap."""

from dataclasses import dataclass
from pathlib import Path
import tiktoken
import yaml

def load_config():
    with open(Path(__file__).parent.parent.parent / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)

@dataclass
class Chunk:
    doc_id: str
    ticker: str
    filing_type: str
    section_id: str
    section_title: str
    chunk_index: int
    content: str
    token_count: int
    overlap_prev: bool
    metadata: dict

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))

def chunk_text(text: str, chunk_size: int, overlap_pct: float,
               encoding_name: str = "cl100k_base") -> list[tuple[str, bool]]:
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    if len(tokens) <= chunk_size:
        return [(text, False)]

    overlap_tokens = int(chunk_size * overlap_pct)
    step = chunk_size - overlap_tokens
    chunks = []

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = enc.decode(chunk_tokens)
        is_overlap = i > 0
        chunks.append((chunk_text, is_overlap))

        if i + chunk_size >= len(tokens):
            break

    return chunks

def chunk_document(parsed_doc, config=None) -> list[Chunk]:
    if config is None:
        config = load_config()["chunking"]

    chunk_size = config["chunk_size"]
    overlap_pct = config["overlap_pct"]
    min_chunk_size = config["min_chunk_size"]
    encoding = config["tokenizer"]
    respect_sections = config["respect_sections"]

    all_chunks = []
    global_index = 0

    for section in parsed_doc.sections:
        text = section.content.strip()
        if not text:
            continue

        token_count = count_tokens(text, encoding)
        if token_count < min_chunk_size:
            continue

        if respect_sections:
            text_chunks = chunk_text(text, chunk_size, overlap_pct, encoding)
        else:
            text_chunks = chunk_text(text, chunk_size, overlap_pct, encoding)

        for chunk_text_str, is_overlap in text_chunks:
            tc = count_tokens(chunk_text_str, encoding)
            if tc < min_chunk_size:
                continue

            all_chunks.append(Chunk(
                doc_id=parsed_doc.doc_id,
                ticker=parsed_doc.ticker,
                filing_type=parsed_doc.filing_type,
                section_id=section.section_id,
                section_title=section.title,
                chunk_index=global_index,
                content=chunk_text_str,
                token_count=tc,
                overlap_prev=is_overlap,
                metadata={
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "has_tables": len(section.tables) > 0
                }
            ))
            global_index += 1

    return all_chunks
