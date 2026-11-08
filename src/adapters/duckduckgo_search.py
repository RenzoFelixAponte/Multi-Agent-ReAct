"""Búsqueda web sin API key. Requiere el paquete `ddgs` (o `duckduckgo_search`)."""
from __future__ import annotations

from domain.evidence import Evidence


class DuckDuckGoSearch:
    def search(self, query: str, k: int = 3) -> list[Evidence]:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        rows = DDGS().text(query, max_results=k)
        return [Evidence(source=r.get("href", "web"), text=r.get("body", "")) for r in rows]
