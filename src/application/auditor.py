from __future__ import annotations

from application.formatting import format_evidence
from application.prompts import AUDIT_PROMPT
from domain.audit_verdict import AuditVerdict
from domain.report import Report
from ports.llm_port import LLMPort


class Auditor:
    """Guardrail: primero reglas deterministas, luego verificación con el LLM."""

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def review(self, report: Report) -> AuditVerdict:
        if not report.answer:
            return AuditVerdict(False, "no hubo respuesta final dentro del límite de pasos")
        if not report.evidence:
            return AuditVerdict(False, "la respuesta no tiene evidencia recuperada")

        out = self._llm.complete(
            AUDIT_PROMPT.format(question=report.question, answer=report.answer,
                                evidence=format_evidence(report.evidence))
        ).strip()
        if out.upper().startswith("APROBADO"):
            return AuditVerdict(True)
        reason = out.split(":", 1)[1].strip() if ":" in out else out
        return AuditVerdict(False, reason or "el auditor no aprobó la respuesta")
