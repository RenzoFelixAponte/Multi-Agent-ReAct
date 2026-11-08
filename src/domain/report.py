from __future__ import annotations

from dataclasses import dataclass

from domain.evidence import Evidence


@dataclass(frozen=True)
class Report:
    """Respuesta del Agente Investigador con la evidencia que la respalda."""

    question: str
    answer: str
    evidence: tuple[Evidence, ...] = ()
