"""Query pipeline: embed question → retrieve chunks → generate answer."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.retrieval.vector_store import VectorStore
from src.utils.ollama_client import OllamaLLM

console = Console()

class RAGPipeline:
    def __init__(self):
        self.store = VectorStore()
        self.llm = OllamaLLM()

    def query(self, question: str, top_k: int = 8, ticker: str = None) -> dict:
        # Retrieve
        results = self.store.search(question, top_k=top_k, ticker=ticker)

        # Build context
        context_parts = []
        for i, r in enumerate(results):
            section = r.get("section_title", "Unknown")
            context_parts.append(
                f"[Chunk {i+1} | Section: {section} | Similarity: {r['similarity']:.3f}]\n{r['content']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Generate
        answer = self.llm.generate(question, context=context)

        return {
            "question": question,
            "answer": answer,
            "retrieval_context": [r["content"] for r in results],
            "chunks": results,
            "context_text": context
        }

@click.command()
@click.argument("question")
@click.option("--top-k", default=8, help="Number of chunks to retrieve")
@click.option("--ticker", default=None, help="Filter by ticker")
@click.option("--show-chunks", is_flag=True, help="Show retrieved chunks")
def main(question, top_k, ticker, show_chunks):
    pipeline = RAGPipeline()

    console.print(f"\n[bold blue]Question:[/bold blue] {question}\n")

    result = pipeline.query(question, top_k=top_k, ticker=ticker)

    # Display answer
    console.print(Panel(result["answer"], title="Answer", border_style="green"))

    # Display retrieval info
    if show_chunks:
        table = Table(title="Retrieved Chunks")
        table.add_column("#", style="dim", width=3)
        table.add_column("Section", width=30)
        table.add_column("Similarity", width=10)
        table.add_column("Preview", width=60)

        for i, chunk in enumerate(result["chunks"]):
            table.add_row(
                str(i + 1),
                chunk.get("section_title", "?")[:30],
                f"{chunk['similarity']:.3f}",
                chunk["content"][:60] + "..."
            )
        console.print(table)

    console.print(f"\n[dim]Chunks retrieved: {len(result['chunks'])} | Top-k: {top_k}[/dim]\n")

if __name__ == "__main__":
    main()
