"""Run DeepEval evaluation on the RAG pipeline with optional chunk deduplication."""

import json
import uuid
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

from src.retrieval.query import RAGPipeline
from src.eval.chunk_dedup import deduplicate_with_stats
from src.utils.db import get_connection

console = Console()

class OllamaEvalModel(DeepEvalBaseLLM):
    """Ollama wrapper for DeepEval's judge model."""

    def __init__(self, model_name="llama3", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        import ollama
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 2048}
        )
        return response["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"ollama/{self.model_name}"

def load_eval_dataset(path: str = None) -> list[dict]:
    if path is None:
        path = Path(__file__).parent / "eval_dataset.json"
    with open(path) as f:
        return json.load(f)

def run_single_eval(question: str, expected_answer: str, pipeline: RAGPipeline,
                    model: OllamaEvalModel, deduplicate: bool = False,
                    dedup_threshold: float = 0.85) -> dict:
    # Query the pipeline
    result = pipeline.query(question)
    retrieval_context = result["retrieval_context"]

    # Optionally deduplicate
    dedup_stats = None
    if deduplicate:
        dedup_result = deduplicate_with_stats(retrieval_context, threshold=dedup_threshold)
        dedup_stats = {
            "original": dedup_result["original_count"],
            "deduped": dedup_result["deduped_count"],
            "removed": dedup_result["removed"],
            "reduction_pct": dedup_result["reduction_pct"]
        }
        retrieval_context = dedup_result["deduped_chunks"]

    # Build test case
    test_case = LLMTestCase(
        input=question,
        actual_output=result["answer"],
        expected_output=expected_answer,
        retrieval_context=retrieval_context,
    )

    # Run metrics
    metrics = {
        "contextual_precision": ContextualPrecisionMetric(threshold=0.5, model=model),
        "contextual_recall": ContextualRecallMetric(threshold=0.5, model=model),
        "faithfulness": FaithfulnessMetric(threshold=0.5, model=model),
        "answer_relevancy": AnswerRelevancyMetric(threshold=0.5, model=model),
    }

    scores = {}
    for name, metric in metrics.items():
        try:
            metric.measure(test_case)
            scores[name] = round(metric.score, 4)
        except Exception as e:
            scores[name] = f"error: {str(e)[:50]}"

    return {
        "question": question,
        "expected_answer": expected_answer,
        "actual_answer": result["answer"],
        "retrieval_context": retrieval_context,
        "scores": scores,
        "dedup_stats": dedup_stats,
        "chunks_used": len(retrieval_context)
    }

def store_eval_results(run_id: str, config: dict, results: list[dict]):
    conn = get_connection()
    cur = conn.cursor()

    # Aggregate metrics
    metric_names = ["contextual_precision", "contextual_recall", "faithfulness", "answer_relevancy"]
    avg_scores = {}
    for metric in metric_names:
        values = [r["scores"].get(metric) for r in results
                  if isinstance(r["scores"].get(metric), (int, float))]
        avg_scores[metric] = round(sum(values) / len(values), 4) if values else None

    cur.execute(
        "INSERT INTO eval_runs (run_id, config, metrics) VALUES (%s, %s, %s)",
        (run_id, json.dumps(config), json.dumps(avg_scores))
    )

    for r in results:
        cur.execute("""
            INSERT INTO eval_results (run_id, question, expected_answer, actual_answer,
                                      retrieval_context, scores)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (run_id, r["question"], r["expected_answer"], r["actual_answer"],
              json.dumps(r["retrieval_context"]), json.dumps(r["scores"])))

    conn.commit()
    cur.close()
    conn.close()

@click.command()
@click.option("--deduplicate", is_flag=True, help="Enable chunk deduplication before eval")
@click.option("--threshold", default=0.85, help="Cosine similarity threshold for dedup")
@click.option("--dataset", default=None, help="Path to eval dataset JSON")
@click.option("--ticker", default=None, help="Filter questions by ticker")
@click.option("--judge-model", default="llama3", help="Ollama model for eval judge")
@click.option("--compare", is_flag=True, help="Run both raw and deduped, show comparison")
def main(deduplicate, threshold, dataset, ticker, judge_model, compare):
    console.print(Panel(
        f"[bold]finrag-eval[/bold] — RAG Evaluation Runner\n"
        f"Judge: ollama/{judge_model} | Dedup: {'ON' if deduplicate else 'OFF'} | Threshold: {threshold}",
        border_style="blue"
    ))

    pipeline = RAGPipeline()
    model = OllamaEvalModel(model_name=judge_model)
    eval_data = load_eval_dataset(dataset)

    if ticker:
        eval_data = [q for q in eval_data if q.get("ticker", "").upper() == ticker.upper()]

    console.print(f"\nRunning {len(eval_data)} evaluation questions...\n")

    if compare:
        # Run both raw and deduped
        raw_results = []
        dedup_results = []

        for i, item in enumerate(eval_data):
            console.print(f"[dim]Q{i+1}/{len(eval_data)}: {item['question'][:60]}...[/dim]")

            raw = run_single_eval(item["question"], item["expected_answer"],
                                  pipeline, model, deduplicate=False)
            raw_results.append(raw)

            dedup = run_single_eval(item["question"], item["expected_answer"],
                                    pipeline, model, deduplicate=True,
                                    dedup_threshold=threshold)
            dedup_results.append(dedup)

        # Comparison table
        table = Table(title="Raw vs Deduped Evaluation Results")
        table.add_column("Metric", style="bold")
        table.add_column("Raw", justify="center")
        table.add_column("Deduped", justify="center")
        table.add_column("Delta", justify="center")

        for metric in ["contextual_precision", "contextual_recall", "faithfulness", "answer_relevancy"]:
            raw_vals = [r["scores"].get(metric) for r in raw_results if isinstance(r["scores"].get(metric), (int, float))]
            ded_vals = [r["scores"].get(metric) for r in dedup_results if isinstance(r["scores"].get(metric), (int, float))]
            raw_avg = sum(raw_vals) / len(raw_vals) if raw_vals else 0
            ded_avg = sum(ded_vals) / len(ded_vals) if ded_vals else 0
            delta = ded_avg - raw_avg
            delta_str = f"[green]+{delta:.3f}[/green]" if delta > 0 else f"[red]{delta:.3f}[/red]"
            table.add_row(metric, f"{raw_avg:.3f}", f"{ded_avg:.3f}", delta_str)

        console.print(table)

        # Store both runs
        run_id_raw = f"raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_id_ded = f"dedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        store_eval_results(run_id_raw, {"deduplicate": False}, raw_results)
        store_eval_results(run_id_ded, {"deduplicate": True, "threshold": threshold}, dedup_results)

    else:
        results = []
        for i, item in enumerate(eval_data):
            console.print(f"[dim]Q{i+1}/{len(eval_data)}: {item['question'][:60]}...[/dim]")
            result = run_single_eval(
                item["question"], item["expected_answer"],
                pipeline, model, deduplicate=deduplicate,
                dedup_threshold=threshold
            )
            results.append(result)

            # Show per-question scores
            scores_str = " | ".join(f"{k}: {v}" for k, v in result["scores"].items()
                                    if isinstance(v, (int, float)))
            console.print(f"  [green]{scores_str}[/green]")
            if result.get("dedup_stats"):
                ds = result["dedup_stats"]
                console.print(f"  [dim]Chunks: {ds['original']} → {ds['deduped']} (-{ds['reduction_pct']:.0f}%)[/dim]")

        # Summary table
        table = Table(title=f"Evaluation Summary ({'Deduped' if deduplicate else 'Raw'})")
        table.add_column("Metric", style="bold")
        table.add_column("Avg Score", justify="center")
        table.add_column("Min", justify="center")
        table.add_column("Max", justify="center")

        for metric in ["contextual_precision", "contextual_recall", "faithfulness", "answer_relevancy"]:
            vals = [r["scores"].get(metric) for r in results if isinstance(r["scores"].get(metric), (int, float))]
            if vals:
                table.add_row(metric, f"{sum(vals)/len(vals):.3f}", f"{min(vals):.3f}", f"{max(vals):.3f}")

        console.print(table)

        run_id = f"{'dedup' if deduplicate else 'raw'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        store_eval_results(run_id, {"deduplicate": deduplicate, "threshold": threshold}, results)
        console.print(f"\n[dim]Results stored: run_id={run_id}[/dim]\n")

if __name__ == "__main__":
    main()
