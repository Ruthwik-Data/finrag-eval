# finrag-eval

A small, local RAG pipeline for financial document analysis with a **focus on evaluation**, built around a 3‑question test on Apple’s FY 2024 SEC filings.

The goal: show how a “simple” finance RAG behaves on concrete questions (revenue, gross margin, R&D), and how it can both refuse to answer and still hallucinate convincing numbers.

---

## What This Does

Given a 10‑K PDF (e.g., Apple’s FY 2024 filing), the pipeline:

```
SEC EDGAR (10-K PDFs)
    → PDF parsing (pdfplumber + section detection)
    → Chunking (section-aware, configurable overlap)
    → Embeddings (Ollama nomic-embed-text)
    → Vector store (Supabase pgvector, local)
    → Retrieval (top-k over pgvector)
    → Answer generation (Ollama llama3)
```

You can then:

- Ask ad‑hoc questions via `src.retrieval.query`.
- Run a tiny 3‑question eval over Apple FY 2024:
  1. Total net revenue vs 2023
  2. Gross margin %
  3. R&D spend and % of revenue

Manual evaluation of those 3 questions is documented in `notes/manual_eval.md`.

---

## Why This Exists

RAG is often demoed with “it answered my question!” screenshots, but very rarely with **clear, labeled evals**.

This repo intentionally keeps things small and opinionated:

- One company (Apple), one filing (10‑K), three concrete, numeric questions.
- A local stack (Ollama + Supabase via Docker) so you can run everything without API keys.
- A manual eval file that shows:
  - Two honest refusals when the numbers aren’t in retrieved context.
  - One confident hallucination with precise but wrong figures.

The idea is to make evaluation behavior visible and tangible, not abstract.

---

## Stack

| Layer            | Tool                                      | Why |
|------------------|-------------------------------------------|-----|
| Document source  | SEC EDGAR (10‑K PDFs)                     | Real, messy financial reports. |
| PDF parsing      | `pdfplumber`                              | Reasonable table + text extraction for filings. |
| Embeddings       | Ollama `nomic-embed-text`                 | Local, free, solid on financial text. |
| Vector store     | Supabase (local via Docker) + `pgvector`  | SQL + vector search in one. |
| LLM              | Ollama `llama3` (or other local models)   | Fully local inference, $0 API cost. |
| Orchestration    | Plain Python scripts                      | Simple, debuggable, no heavy framework. |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for local Supabase)
- [Ollama](https://ollama.com) installed with models pulled

### 1. Setup

```bash
git clone https://github.com/YOUR_USERNAME/finrag-eval.git
cd finrag-eval

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull Ollama models
ollama pull nomic-embed-text
ollama pull llama3
```

### 2. Start Local Supabase

```bash
docker compose up -d
```

### 3. Initialize Database

```bash
python scripts/init_db.py
```

### 4. Download & Ingest Apple’s 10‑K

```bash
# Download Apple's latest 10-K from SEC EDGAR
python -m src.ingestion.edgar_download --ticker AAPL --filing-type 10-K

# Parse, chunk, and embed
python -m src.ingestion.ingest --input data/raw/AAPL_10K.pdf
```

### 5. Query the Pipeline

Ask any question:

```bash
python -m src.retrieval.query "What was Apple's total net revenue for fiscal year 2024, and how did it change compared to fiscal year 2023?"
```

(For the other two eval questions, see `src/eval/eval_dataset.json`.)

---

## Tiny Evaluation Suite

The repo includes a minimal eval dataset at:

- `src/eval/eval_dataset.json` – three labeled Q&A pairs on Apple FY 2024:
  1. Total net revenue and change vs 2023.
  2. Gross margin percentage.
  3. R&D spend and % of revenue.

You can manually run each question through the pipeline:

```bash
python -m src.retrieval.query "QUESTION_HERE"
```

Then compare against the expected answers.

The file `notes/manual_eval.md` captures:

- The exact model answers returned by the RAG pipeline.
- The expected ground‑truth values (from earnings reports / filings).
- A verdict for each question (refusal vs hallucination).

---

## Project Structure

```text
finrag-eval/
├── configs/
│   └── pipeline.yaml          # Pipeline parameters
├── src/
│   ├── ingestion/
│   │   ├── edgar_download.py   # SEC EDGAR 10-K/10-Q downloader
│   │   ├── pdf_parser.py       # PDF → structured sections
│   │   ├── chunker.py          # Section-aware chunking with configurable overlap
│   │   └── ingest.py           # End-to-end ingestion
│   ├── retrieval/
│   │   ├── embedder.py         # Ollama embedding wrapper
│   │   ├── vector_store.py     # Supabase pgvector operations
│   │   └── query.py            # Retrieve + generate answer
│   ├── eval/
│   │   ├── run_eval.py         # (Optional) automated eval hooks
│   │   ├── eval_dataset.json   # Ground truth Q&A pairs
│   │   └── metrics.py          # Placeholder for custom metrics
│   └── utils/
│       ├── ollama_client.py    # Ollama client
│       └── db.py               # Supabase connection helper
├── notes/
│   └── manual_eval.md          # Manual evaluation of the 3 Apple FY24 questions
├── scripts/
│   └── init_db.py              # Database + pgvector setup
├── tests/
│   └── test_chunker.py
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

---

## What This Shows

This repo is intentionally small but opinionated:

- **Evaluation matters:** even a tiny 3‑Q test can reveal both:
  - *Honest refusals* when the answer isn’t in retrieved context.
  - *Confident hallucinations* with precise but wrong numbers.
- **Local is enough:** you can explore RAG eval behavior without any external APIs.
- **Extendable:** swap in more filings, add more questions, or plug in your favorite eval framework on top of the same ingestion + retrieval stack.

---

## License

MIT
```