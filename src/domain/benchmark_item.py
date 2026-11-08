from __future__ import annotations

from dataclasses import dataclass

from domain.evidence import Evidence


@dataclass(frozen=True)
class BenchmarkItem:
    """Consulta de evaluación con su respuesta de referencia y el contexto que la respalda."""

    dataset: str
    item_id: str
    question: str
    reference: str
    context: tuple[Evidence, ...] = ()
