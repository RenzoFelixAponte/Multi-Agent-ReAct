"""Baseline de comparación: RAG tradicional de un solo paso, sin agentes ni auditoría."""
from __future__ import annotations

from ports.ports import KnowledgeBasePort, LLMPort

NAIVE_PROMPT = """Responde usando solo el contexto.
Contexto:
{context}

Pregunta: {question}
Respuesta:"""


def naive_rag(question: str, llm: LLMPort, kb: KnowledgeBasePort, k: int = 3) -> str:
    context = "\n".join(f"[{e.source}] {e.text}" for e in kb.retrieve(question, k))
    return llm.complete(NAIVE_PROMPT.format(context=context, question=question)).strip()
