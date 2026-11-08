"""Casos de uso de los agentes. Dependen solo de puertos, nunca de adaptadores."""
from __future__ import annotations

import re

from proyecto2.domain.models import AuditVerdict, Evidence, Report
from proyecto2.ports.ports import KnowledgeBasePort, LLMPort, WebSearchPort

MAX_REACT_STEPS = 4

ACTION_RE = re.compile(r"Action:\s*(kb|web)\[(.*?)\]", re.S)
FINAL_RE = re.compile(r"Final:\s*(.*)", re.S)

REACT_PROMPT = """Responde la pregunta usando el formato:
Thought: ...
Action: kb[consulta] o web[consulta]
Observation: (lo escribe el sistema)
... repite ...
Final: respuesta final

Pregunta: {question}
Retroalimentación previa del auditor: {feedback}
{scratch}"""

AUDIT_PROMPT = """Eres el Auditor de Hechos. Verifica si la respuesta está respaldada por la evidencia.
Responde solo "APROBADO" o "RECHAZADO: motivo".

Pregunta: {question}
Respuesta: {answer}
Evidencia:
{evidence}"""

REDACT_PROMPT = """Redacta una respuesta clara y citada en base a la respuesta verificada.
Pregunta: {question}
Respuesta verificada: {answer}
Fuentes: {sources}"""


def _format_evidence(evidence: tuple[Evidence, ...] | list[Evidence]) -> str:
    return "\n".join(f"[{e.source}] {e.text}" for e in evidence) or "(sin evidencia)"


class Investigator:
    """Bucle ReAct: Thought -> Action -> Observation hasta obtener Final."""

    def __init__(self, llm: LLMPort, web: WebSearchPort, kb: KnowledgeBasePort,
                 max_steps: int = MAX_REACT_STEPS) -> None:
        self._llm = llm
        self._web = web
        self._kb = kb
        self._max_steps = max_steps

    def run(self, question: str, feedback: str = "") -> Report:
        evidence: list[Evidence] = []
        scratch = ""
        for _ in range(self._max_steps):
            out = self._llm.complete(
                REACT_PROMPT.format(question=question, feedback=feedback or "ninguna", scratch=scratch)
            ).strip()
            scratch += out + "\n"

            final = FINAL_RE.search(out)
            if final:
                return Report(question, final.group(1).strip(), tuple(evidence))

            action = ACTION_RE.search(out)
            if action is None:
                scratch += "Observation: formato no válido. Usa Action o Final.\n"
                continue

            tool, arg = action.group(1), action.group(2).strip()
            results = self._kb.retrieve(arg) if tool == "kb" else self._web.search(arg)
            evidence.extend(results)
            scratch += f"Observation: {_format_evidence(results) if results else 'sin resultados'}\n"

        return Report(question, "", tuple(evidence))


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
                                evidence=_format_evidence(report.evidence))
        ).strip()
        if out.upper().startswith("APROBADO"):
            return AuditVerdict(True)
        reason = out.split(":", 1)[1].strip() if ":" in out else out
        return AuditVerdict(False, reason or "el auditor no aprobó la respuesta")


class Redactor:
    """Produce el texto final a partir de una respuesta ya auditada."""

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def write(self, report: Report) -> str:
        sources = ", ".join(sorted({e.source for e in report.evidence})) or "ninguna"
        return self._llm.complete(
            REDACT_PROMPT.format(question=report.question, answer=report.answer, sources=sources)
        ).strip()
