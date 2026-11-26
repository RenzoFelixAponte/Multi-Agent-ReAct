from __future__ import annotations

from application.answer_checks import has_code_block
from application.language import question_language
from application.prompts import REDACT_PROMPT, REDACT_RULES
from domain.report import Report
from ports.llm_port import LLMPort


class Redactor:
    """Produce el texto final a partir de una respuesta ya auditada, sin agregarle contenido."""

    def __init__(self, llm: LLMPort, task: str = "general") -> None:
        self._llm = llm
        self._rule = REDACT_RULES[task]

    def write(self, report: Report) -> str:
        # El código ya fue ejecutado y auditado: reescribirlo con un LLM solo puede estropearlo.
        if has_code_block(report.answer):
            return report.answer
        return self._llm.complete(
            REDACT_PROMPT.format(question=report.question, answer=report.answer, format_rule=self._rule,
                                 language=question_language(report.question))
        ).strip()
