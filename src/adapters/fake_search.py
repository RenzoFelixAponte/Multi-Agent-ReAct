"""Buscador web falso para pruebas: siempre devuelve la misma lista."""
from __future__ import annotations

from domain.evidence import Evidence


class StaticSearch:
    def __init__(self, results: list[Evidence]) -> None:
        self._results = results

    def search(self, query: str, k: int = 3) -> list[Evidence]:
        return self._results[:k]
