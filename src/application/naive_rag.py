"""Baseline de comparación: RAG tradicional de un solo paso, sin agentes ni auditoría."""
from __future__ import annotations

from application.formatting import format_evidence
from application.language import question_language
from application.prompts import NAIVE_PROMPT
from domain.evidence import Evidence
from ports.knowledge_base_port import KnowledgeBasePort
from ports.llm_port import LLMPort


def naive_rag_with_context(question: str, llm: LLMPort, kb: KnowledgeBasePort,
                           k: int = 3) -> tuple[str, list[Evidence]]:
    """Una recuperación y una llamada al LLM. Devuelve también el contexto, que luego juzga la fidelidad."""
    evidence = kb.retrieve(question, k)
    prompt = NAIVE_PROMPT.format(context=format_evidence(evidence), question=question,
                                 language=question_language(question))
    return llm.complete(prompt).strip(), evidence


def naive_rag(question: str, llm: LLMPort, kb: KnowledgeBasePort, k: int = 3) -> str:
    return naive_rag_with_context(question, llm, kb, k)[0]
