from __future__ import annotations

from dataclasses import dataclass

from domain.evidence import Evidence


@dataclass(frozen=True)
class Report:
    """Respuesta del Agente Investigador con la evidencia que la respalda.

    `trace` es el bucle ReAct de la ronda (Thought, Action y Observation), para mostrarlo y auditarlo.
    """

    question: str
    answer: str
    evidence: tuple[Evidence, ...] = ()
    trace: str = ""
