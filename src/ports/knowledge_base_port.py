from __future__ import annotations

from typing import Protocol

from domain.evidence import Evidence


class KnowledgeBasePort(Protocol):
    """Contrato para la base de conocimiento vectorial."""

    def retrieve(self, query: str, k: int = 3) -> list[Evidence]: ...
