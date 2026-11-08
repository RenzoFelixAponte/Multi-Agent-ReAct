from __future__ import annotations

from typing import Protocol

from domain.evidence import Evidence


class WebSearchPort(Protocol):
    """Contrato para búsqueda en la web."""

    def search(self, query: str, k: int = 3) -> list[Evidence]: ...
