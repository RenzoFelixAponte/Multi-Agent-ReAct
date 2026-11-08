from __future__ import annotations

import re

from application.formatting import format_evidence
from application.prompts import REACT_PROMPT
from domain.evidence import Evidence
from domain.report import Report
from ports.knowledge_base_port import KnowledgeBasePort
from ports.llm_port import LLMPort
from ports.web_search_port import WebSearchPort

MAX_REACT_STEPS = 4

ACTION_RE = re.compile(r"Action:\s*(kb|web)\[(.*?)\]", re.S)
FINAL_RE = re.compile(r"Final:\s*(.*)", re.S)


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
            raw = self._llm.complete(
                REACT_PROMPT.format(question=question, feedback=feedback or "ninguna", scratch=scratch)
            ).strip()
            # El modelo puede inventar su propia Observation. Solo el sistema escribe observaciones,
            # así que se descarta todo lo que venga después de la primera Observation del modelo.
            out = raw.split("Observation:", 1)[0].strip()
            scratch += out + "\n"

            action = ACTION_RE.search(out)
            if action is None:
                final = FINAL_RE.search(out)
                if final:
                    return Report(question, final.group(1).strip(), tuple(evidence))
                scratch += "Observation: formato no válido. Usa Action o Final.\n"
                continue

            tool, arg = action.group(1), action.group(2).strip()
            results = self._kb.retrieve(arg) if tool == "kb" else self._web.search(arg)
            evidence.extend(results)
            scratch += f"Observation: {format_evidence(results) if results else 'sin resultados'}\n"

        return Report(question, "", tuple(evidence))
