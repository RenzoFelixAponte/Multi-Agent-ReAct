from __future__ import annotations

from application.answer_checks import code_was_executed, extract_asserts, extract_code, has_code_block
from application.formatting import format_evidence
from application.prompts import AUDIT_PROMPT, AUDIT_RULES
from domain.audit_verdict import AuditVerdict
from domain.evidence import Evidence
from domain.report import Report
from ports.code_runner_port import CodeRunnerPort
from ports.llm_port import LLMPort

CODE_TIMEOUT = 10.0
MAX_ERROR_CHARS = 600


class Auditor:
    """Guardrail: primero reglas deterministas, luego verificación con el LLM.

    Con `runner`, el código de una respuesta se audita ejecutándolo junto con los `assert` del
    enunciado: es una verificación más fuerte que pedirle a un LLM que lo lea.
    """

    def __init__(self, llm: LLMPort, runner: CodeRunnerPort | None = None, task: str = "general") -> None:
        self._llm = llm
        self._runner = runner
        self._rule = AUDIT_RULES[task]

    def review(self, report: Report) -> AuditVerdict:
        if not report.answer:
            return AuditVerdict(False, "no hubo respuesta final dentro del límite de pasos")
        tests = extract_asserts(report.question)
        if self._runner is not None and tests:
            return self._run_tests(report.answer, tests)
        if not report.evidence:
            return AuditVerdict(False, "la respuesta no tiene evidencia: usa las herramientas para respaldarla")
        if has_code_block(report.answer) and not code_was_executed(report.answer, report.evidence):
            return AuditVerdict(False, "la respuesta incluye código que no se ejecutó tal cual con la herramienta "
                                       "python; entrega solo código ejecutado y verificado con sus assert")
        out = self._llm.complete(
            AUDIT_PROMPT.format(question=report.question, answer=report.answer, task_rule=self._rule,
                                evidence=format_evidence(report.evidence))
        ).strip()
        if out.upper().startswith("APROBADO"):
            return AuditVerdict(True)
        reason = out.split(":", 1)[1].strip() if ":" in out else out
        return AuditVerdict(False, reason or "el auditor no aprobó la respuesta")

    def _run_tests(self, answer: str, tests: list[str]) -> AuditVerdict:
        if not has_code_block(answer):
            return AuditVerdict(False, "la respuesta debe traer el código completo en un bloque ```python")
        program = "\n".join([extract_code(answer), *tests])
        result = self._runner.run(program, CODE_TIMEOUT)
        if result.ok:
            # La ejecución respalda la respuesta aunque el Investigador no la haya ejecutado en su ronda.
            check = Evidence("auditor", f"código entregado ejecutado con las pruebas del enunciado:\n{program}\n"
                                        "salida: todas las pruebas pasaron")
            return AuditVerdict(True, evidence=(check,))
        return AuditVerdict(False, "el código entregado no pasa las pruebas del enunciado. "
                                   f"Error: {result.output[-MAX_ERROR_CHARS:]}")
