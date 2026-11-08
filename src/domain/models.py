"""Modelos de dominio. No dependen de ningún framework ni de ningún proveedor."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    """Fragmento de contexto recuperado (de la base vectorial o de la web)."""

    source: str
    text: str


@dataclass(frozen=True)
class Report:
    """Respuesta del Agente Investigador con la evidencia que la respalda."""

    question: str
    answer: str
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class BenchmarkItem:
    """Consulta de evaluación con su respuesta de referencia y el contexto que la respalda."""

    dataset: str
    item_id: str
    question: str
    reference: str
    context: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class AuditVerdict:
    """Veredicto del Agente Auditor. Si no está aprobado, `reason` guía el siguiente intento."""

    approved: bool
    reason: str = ""
