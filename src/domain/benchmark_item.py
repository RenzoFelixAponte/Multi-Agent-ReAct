from __future__ import annotations

from dataclasses import dataclass

from domain.evidence import Evidence


@dataclass(frozen=True)
class BenchmarkItem:
    """Consulta de evaluación con su respuesta de referencia y el contexto que la respalda.

    `tests` y `setup_code` solo se usan en MBPP: son los `assert` que verifican el código generado.
    """

    dataset: str
    item_id: str
    question: str
    reference: str
    context: tuple[Evidence, ...] = ()
    tests: tuple[str, ...] = ()
    setup_code: str = ""
