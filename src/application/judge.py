"""LLM-as-a-Judge con las definiciones de RAGAS: Faithfulness y Answer Relevance."""
from __future__ import annotations

import re
from collections.abc import Sequence

from application.formatting import format_evidence
from application.prompts import FAITHFULNESS_PROMPT, RELEVANCE_PROMPT
from domain.evidence import Evidence
from ports.llm_port import LLMPort

MAX_CONTEXT_CHARS = 6000
MAX_ANSWER_CHARS = 3000
ATTEMPTS = 2

# Mistral a veces ignora el formato pedido y responde "1. afirmación (YES) - explicación"
_LEADING = re.compile(r"^[\s\-\*\d\.\)]*(SI|SÍ|YES|NO)\s*:", re.I)
_PARENTHESIZED = re.compile(r"\((SI|SÍ|YES|NO)\)", re.I)
_SCORE = re.compile(r"PUNTAJE\s*:\s*(\d{1,2})", re.I)
_ANY_INT = re.compile(r"\b(\d{1,2})\b")


def parse_faithfulness(text: str) -> float | None:
    """Fracción de afirmaciones marcadas SI. None si el juez no devolvió ninguna con el formato pedido."""
    verdicts = []
    for line in text.splitlines():
        match = _LEADING.match(line) or _PARENTHESIZED.search(line)
        if match:
            verdicts.append(match.group(1).upper())
    if not verdicts:
        return None
    return sum(v != "NO" for v in verdicts) / len(verdicts)


def parse_relevance(text: str) -> float | None:
    """Puntaje de 0 a 10 normalizado a 0-1. None si no hay un entero válido."""
    match = _SCORE.search(text) or _ANY_INT.search(text)
    if match is None:
        return None
    value = int(match.group(1))
    return value / 10 if 0 <= value <= 10 else None


class Judge:
    """Dos métricas sin referencia: la fidelidad se mide contra el contexto y la relevancia contra la pregunta."""

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def faithfulness(self, answer: str, evidence: Sequence[Evidence], question: str = "") -> float | None:
        """Los datos del enunciado cuentan como información dada: en GSM8K y MBPP el contexto son
        solo salidas de herramientas, y la respuesta repite con razón los datos de la pregunta."""
        if not evidence:
            return 0.0  # nada en el contexto puede respaldar la respuesta
        prompt = FAITHFULNESS_PROMPT.format(
            question=question, context=format_evidence(evidence)[:MAX_CONTEXT_CHARS],
            answer=answer[:MAX_ANSWER_CHARS])
        return self._ask(prompt, parse_faithfulness)

    def relevance(self, question: str, answer: str) -> float | None:
        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer[:MAX_ANSWER_CHARS])
        return self._ask(prompt, parse_relevance)

    def _ask(self, prompt: str, parse) -> float | None:
        for _ in range(ATTEMPTS):
            score = parse(self._llm.complete(prompt))
            if score is not None:
                return score
        return None
