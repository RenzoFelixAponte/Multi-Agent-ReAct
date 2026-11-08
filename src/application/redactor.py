from __future__ import annotations

from application.prompts import REDACT_PROMPT
from domain.report import Report
from ports.llm_port import LLMPort


class Redactor:
    """Produce el texto final a partir de una respuesta ya auditada."""

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def write(self, report: Report) -> str:
        sources = ", ".join(sorted({e.source for e in report.evidence})) or "ninguna"
        return self._llm.complete(
            REDACT_PROMPT.format(question=report.question, answer=report.answer, sources=sources)
        ).strip()
