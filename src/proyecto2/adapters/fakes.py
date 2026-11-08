"""Adaptadores falsos para pruebas unitarias. No requieren GPU, red ni claves."""
from __future__ import annotations

from proyecto2.domain.models import Evidence


class ScriptedLLM:
    """Devuelve respuestas en orden y guarda los prompts recibidos."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("ScriptedLLM sin respuestas restantes")
        return self._responses.pop(0)


class StaticSearch:
    """Buscador web falso: siempre devuelve la misma lista."""

    def __init__(self, results: list[Evidence]) -> None:
        self._results = results

    def search(self, query: str, k: int = 3) -> list[Evidence]:
        return self._results[:k]


class InMemoryKB:
    """Base de conocimiento en memoria con coincidencia por palabras compartidas."""

    def __init__(self, documents: list[Evidence]) -> None:
        self._documents = documents

    def retrieve(self, query: str, k: int = 3) -> list[Evidence]:
        terms = set(query.lower().split())
        scored = sorted(
            self._documents,
            key=lambda d: len(terms & set(d.text.lower().split())),
            reverse=True,
        )
        return [d for d in scored if terms & set(d.text.lower().split())][:k]
