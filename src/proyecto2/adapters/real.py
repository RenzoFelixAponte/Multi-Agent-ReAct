"""Adaptadores reales. Las dependencias pesadas se importan al instanciar, no al cargar el módulo.

Pendiente de validar en Colab con T4: MistralLocalLLM y ChromaKB.
"""
from __future__ import annotations

from proyecto2.domain.models import Evidence


class MistralLocalLLM:
    """LLM local cuantizado en 4 bits (bitsandbytes) para la T4."""

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3", max_new_tokens: int = 512) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant, device_map="auto"
        )
        self._max_new_tokens = max_new_tokens

    def complete(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        ids = self._tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        ids = ids.to(self._model.device)
        out = self._model.generate(ids, max_new_tokens=self._max_new_tokens, do_sample=False)
        return self._tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


class DuckDuckGoSearch:
    """Búsqueda web sin API key. Requiere el paquete `ddgs` (o `duckduckgo_search`)."""

    def search(self, query: str, k: int = 3) -> list[Evidence]:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        rows = DDGS().text(query, max_results=k)
        return [Evidence(source=r.get("href", "web"), text=r.get("body", "")) for r in rows]


class ChromaKB:
    """Base vectorial local con ChromaDB."""

    def __init__(self, collection_name: str = "proyecto2", path: str = "./chroma_db") -> None:
        import chromadb

        client = chromadb.PersistentClient(path=path)
        self._col = client.get_or_create_collection(collection_name)

    def index(self, documents: list[Evidence]) -> None:
        self._col.upsert(
            ids=[f"doc-{i}" for i in range(len(documents))],
            documents=[d.text for d in documents],
            metadatas=[{"source": d.source} for d in documents],
        )

    def retrieve(self, query: str, k: int = 3) -> list[Evidence]:
        res = self._col.query(query_texts=[query], n_results=k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return [Evidence(source=m.get("source", "kb"), text=t) for t, m in zip(docs, metas)]
