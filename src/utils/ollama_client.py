"""Ollama client for embeddings and generation."""

import ollama
import yaml
from pathlib import Path

def load_config():
    with open(Path(__file__).parent.parent.parent / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)

class OllamaEmbedder:
    def __init__(self, model=None):
        cfg = load_config()["embedding"]
        self.model = model or cfg["model"]
        self.batch_size = cfg["batch_size"]

    def embed(self, text: str) -> list[float]:
        response = ollama.embed(model=self.model, input=text)
        return response["embeddings"][0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = ollama.embed(model=self.model, input=batch)
            embeddings.extend(response["embeddings"])
        return embeddings

class OllamaLLM:
    def __init__(self, model=None):
        cfg = load_config()["generation"]
        self.model = model or cfg["model"]
        self.temperature = cfg["temperature"]
        self.max_tokens = cfg["max_tokens"]
        self.system_prompt = cfg["system_prompt"]

    def generate(self, prompt: str, context: str = "") -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})
        response = ollama.chat(
            model=self.model, messages=messages,
            options={"temperature": self.temperature, "num_predict": self.max_tokens}
        )
        return response["message"]["content"]
