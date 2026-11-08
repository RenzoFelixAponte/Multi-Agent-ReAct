from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditVerdict:
    """Veredicto del Agente Auditor. Si no está aprobado, `reason` guía el siguiente intento."""

    approved: bool
    reason: str = ""
