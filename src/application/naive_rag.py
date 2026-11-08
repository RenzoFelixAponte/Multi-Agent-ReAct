"""Baseline de comparación: RAG tradicional de un solo paso, sin agentes ni auditoría."""
from __future__ import annotations

from application.prompts import NAIVE_PROMPT
from ports.knowledge_base_port import KnowledgeBasePort
from ports.llm_port import LLMPort


def naive_rag(question: str, llm: LLMPort, kb: KnowledgeBasePort, k: int = 3) -> str:
    context = "\n".join(f"[{e.source}] {e.text}" for e in kb.retrieve(question, k))
    return llm.complete(NAIVE_PROMPT.format(context=context, question=question)).strip()
