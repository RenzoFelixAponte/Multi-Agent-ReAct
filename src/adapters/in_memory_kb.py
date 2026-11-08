"""Base de conocimiento en memoria para pruebas, con coincidencia por palabras compartidas."""
from __future__ import annotations

from domain.evidence import Evidence


class InMemoryKB:
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
