"""Parse financial PDFs into structured sections with table preservation."""

import re
from dataclasses import dataclass, field
from pathlib import Path
import pdfplumber
import yaml

def load_config():
    with open(Path(__file__).parent.parent.parent / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)

@dataclass
class Section:
    section_id: str
    title: str
    content: str
    page_start: int
    page_end: int
    tables: list = field(default_factory=list)

@dataclass
class ParsedDocument:
    doc_id: str
    ticker: str
    filing_type: str
    total_pages: int
    sections: list[Section]
    raw_text: str

def detect_section(text: str, markers: list[str]) -> tuple[str, str] | None:
    for marker in markers:
        pattern = rf"^({re.escape(marker)})\s*[-—.]?\s*(.+?)$"
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            section_id = marker.strip().rstrip(".")
            title = match.group(2).strip()
            return section_id, title
    return None

def extract_tables(page) -> list[dict]:
    tables = []
    try:
        for table in page.extract_tables():
            if table and len(table) > 1:
                # Clean cells
                cleaned = []
                for row in table:
                    cleaned.append([cell.strip() if cell else "" for cell in row])
                # Convert to markdown-style table
                header = " | ".join(cleaned[0])
                separator = " | ".join(["---"] * len(cleaned[0]))
                rows = "\n".join(" | ".join(row) for row in cleaned[1:])
                tables.append({
                    "markdown": f"{header}\n{separator}\n{rows}",
                    "raw": cleaned
                })
    except Exception:
        pass
    return tables

def parse_pdf(filepath: str, ticker: str = "", filing_type: str = "10-K") -> ParsedDocument:
    cfg = load_config()["parsing"]
    markers = cfg["section_markers"]
    path = Path(filepath)
    doc_id = path.stem

    sections = []
    current_section = None
    current_content = []
    current_tables = []
    current_page_start = 0
    all_text = []

    with pdfplumber.open(filepath) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            all_text.append(text)

            # Extract tables from this page
            if cfg["detect_tables"]:
                page_tables = extract_tables(page)
            else:
                page_tables = []

            if cfg["detect_sections"]:
                lines = text.split("\n")
                for line in lines:
                    detected = detect_section(line, markers)
                    if detected:
                        # Save previous section
                        if current_section:
                            sections.append(Section(
                                section_id=current_section[0],
                                title=current_section[1],
                                content="\n".join(current_content),
                                page_start=current_page_start,
                                page_end=page_num,
                                tables=current_tables
                            ))
                        current_section = detected
                        current_content = []
                        current_tables = []
                        current_page_start = page_num
                    else:
                        current_content.append(line)

                current_tables.extend(page_tables)
            else:
                current_content.append(text)
                current_tables.extend(page_tables)

        # Save last section
        if current_section:
            sections.append(Section(
                section_id=current_section[0],
                title=current_section[1],
                content="\n".join(current_content),
                page_start=current_page_start,
                page_end=total_pages - 1,
                tables=current_tables
            ))
        elif current_content:
            # No sections detected — treat entire doc as one section
            sections.append(Section(
                section_id="full_doc",
                title="Full Document",
                content="\n".join(current_content),
                page_start=0,
                page_end=total_pages - 1,
                tables=current_tables
            ))

    # Inject table markdown into section content
    for section in sections:
        for table in section.tables:
            section.content += f"\n\n[TABLE]\n{table['markdown']}\n[/TABLE]\n"

    return ParsedDocument(
        doc_id=doc_id, ticker=ticker, filing_type=filing_type,
        total_pages=total_pages, sections=sections,
        raw_text="\n\n".join(all_text)
    )

def parse_html_filing(filepath: str, ticker: str = "", filing_type: str = "10-K") -> ParsedDocument:
    """Parse HTML/HTM SEC filings (most modern 10-Ks are HTML, not PDF)."""
    from html.parser import HTMLParser
    import html

    path = Path(filepath)
    raw = path.read_text(encoding="utf-8", errors="ignore")

    # Strip HTML tags, keep text
    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text = []
            self.skip = False
        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self.skip = True
        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self.skip = False
        def handle_data(self, data):
            if not self.skip:
                self.text.append(data)

    extractor = TextExtractor()
    extractor.feed(raw)
    text = " ".join(extractor.text)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    cfg = load_config()["parsing"]
    markers = cfg["section_markers"]

    # Split by section markers
    sections = []
    parts = re.split(r'(Item \d+[A-B]?\.)', text, flags=re.IGNORECASE)

    for i in range(1, len(parts) - 1, 2):
        section_id = parts[i].strip().rstrip(".")
        content = parts[i + 1].strip()[:50000]  # Cap per section
        title_match = re.match(r'^[\s.—-]*(.+?)(?:\n|$)', content)
        title = title_match.group(1).strip()[:100] if title_match else section_id
        sections.append(Section(
            section_id=section_id, title=title,
            content=content, page_start=0, page_end=0, tables=[]
        ))

    if not sections:
        sections.append(Section(
            section_id="full_doc", title="Full Document",
            content=text[:200000], page_start=0, page_end=0, tables=[]
        ))

    return ParsedDocument(
        doc_id=path.stem, ticker=ticker, filing_type=filing_type,
        total_pages=0, sections=sections, raw_text=text[:200000]
    )
