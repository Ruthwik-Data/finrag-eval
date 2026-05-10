# finrag-eval

A local RAG evaluation pipeline on Apple's FY 2024 10-K filing — built to understand where financial document retrieval breaks down, not just to make a demo that works.

## Problem & Why

Financial analysts and AI teams building RAG over SEC filings face a dangerous failure mode: the system answers confidently even when it shouldn't. Standard RAG evals don't catch the difference between "I don't know" (good) and "Revenue was $387.2B" when the actual number is $383.3B (dangerous). This project measures that gap on real data.

## What This Is

An end-to-end local RAG pipeline — SEC EDGAR ingestion through answer generation — with a 3-question evaluation suite grounded against Apple's actual FY 2024 10-K figures.

## Architecture

```
SEC EDGAR (10-K PDF)
    → pdfplumber (section-aware PDF parsing)
    → Chunker (configurable overlap, section boundaries preserved)
    → Ollama nomic-embed-text (local embeddings)
    → Supabase pgvector / Docker (local vector store)
    → Top-k retrieval
    → Ollama llama3 (local LLM, $0 API cost)
    → Evaluation layer (DeepEval metrics + manual ground-truth check)
```

| Layer | Tool | Why |
|-------|------|-----|
| PDF parsing | pdfplumber | Handles financial table extraction reasonably well |
| Embeddings | nomic-embed-text | Local, free, strong on financial terminology |
| Vector store | Supabase + pgvector | SQL + vector in one; production-representative |
| LLM | llama3 via Ollama | Fully local — no API costs, reproducible |
| Orchestration | Plain Python | Debuggable; no hidden abstractions |

## Evaluation & Results

3-question test suite grounded against Apple's FY 2024 10-K (ground truth pulled directly from filings):

| Question | Ground Truth | System Response | Verdict |
|----------|-------------|-----------------|--------|
| Total net revenue vs FY 2023? | $391.0B (+2% YoY) | Refused — "not found in context" | ✅ Honest refusal |
| Gross margin %? | 46.2% | Refused — "not found in context" | ✅ Honest refusal |
| R&D spend + % of revenue? | $31.4B / 8.0% | Answered confidently with wrong figures | ❌ Confident hallucination |

**Key finding: 2/3 honest refusals. 1/3 confident hallucination with precise but incorrect numbers.**

The dangerous failure mode is not "I don't know" — it's "The answer is X" where X is wrong and sounds credible.

## What This Led To

Running DeepEval's `ContextualPrecisionMetric` on this pipeline exposed a metric-level bug: overlapping chunks (10-20% overlap, standard for preserving table/section boundaries) were being penalized as independent retrieval failures — making eval scores *worse* as chunk quality improved.

This became GitHub Issue [#2594](https://github.com/confident-ai/deepeval/issues/2594) on the DeepEval repo. The Confident AI team is shipping the fix (`group_by` parameter) in the next release.

**The eval found a bug in the eval framework. That's the point.**

## How to Use

```bash
git clone https://github.com/Ruthwik-Data/finrag-eval
cd finrag-eval
docker-compose up -d
python scripts/init_db.py
python src/ingestion/ingest.py
python src/retrieval/query.py
python src/eval/metrics.py
```

Results and manual verdicts in `notes/manual_eval.md`.

## Lessons Learned

1. **Confident hallucination is worse than refusal.** A system that says "I don't know" is safer than one that gives a precise wrong number. Calibrating refusal behavior is a product decision, not just a technical one.
2. **Overlap helps retrieval, hurts naive evals.** Increasing chunk overlap improved answer grounding but lowered DeepEval precision scores — because the metric penalized redundant chunks as misses. Evaluation metrics can lie about retrieval quality.
3. **Local-first forced honesty.** Running fully locally ($0 API cost) meant I couldn't rely on GPT-4 to paper over weak retrieval. The results are less polished but more honest.

## Known Limitations

- 3-question eval is illustrative, not statistically significant
- Ground truth extracted manually — possible human error on edge cases
- llama3 locally is weaker than GPT-4 class models
- No automated hallucination detection — verdicts are manual
- Section detection in pdfplumber is heuristic and may miss boundaries in complex filings
