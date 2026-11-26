from __future__ import annotations

from dataclasses import dataclass

from domain.evidence import Evidence


@dataclass(frozen=True)
class AuditVerdict:
    """Veredicto del Agente Auditor. Si no está aprobado, `reason` guía el siguiente intento.

    `evidence` es lo que el Auditor verificó por su cuenta (por ejemplo, ejecutar el código con las
    pruebas del enunciado). Se suma a la evidencia del reporte aprobado.
    """

    approved: bool
    reason: str = ""
    evidence: tuple[Evidence, ...] = ()
