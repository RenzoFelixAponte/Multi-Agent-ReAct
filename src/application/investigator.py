from __future__ import annotations

import re

from application.answer_checks import extract_code
from application.calculator import calculate
from application.formatting import format_evidence
from application.prompts import REACT_PROMPT, TOOL_DESCRIPTIONS
from domain.evidence import Evidence
from domain.report import Report
from ports.code_runner_port import CodeRunnerPort
from ports.knowledge_base_port import KnowledgeBasePort
from ports.llm_port import LLMPort
from ports.web_search_port import WebSearchPort

MAX_REACT_STEPS = 6
CODE_TIMEOUT = 5.0

ACTION_START_RE = re.compile(r"Action:\s*(kb|web|calc|python)\[")
NEXT_SECTION_RE = re.compile(r"\n\s*(?:Thought|Action|Observation|Final):")
FINAL_RE = re.compile(r"Final:\s*(.*)", re.S)


def parse_action(text: str) -> tuple[str, str] | None:
    """Extrae (herramienta, argumento). El argumento llega hasta el último `]` de su sección,
    así el código o las expresiones pueden contener corchetes."""
    start = ACTION_START_RE.search(text)
    if start is None:
        return None
    rest = text[start.end():]
    nxt = NEXT_SECTION_RE.search(rest)
    if nxt:
        rest = rest[: nxt.start()]
    close = rest.rfind("]")
    return start.group(1), (rest[:close] if close != -1 else rest).strip()


class Investigator:
    """Bucle ReAct: Thought -> Action -> Observation hasta obtener Final."""

    def __init__(self, llm: LLMPort, web: WebSearchPort, kb: KnowledgeBasePort,
                 max_steps: int = MAX_REACT_STEPS, runner: CodeRunnerPort | None = None) -> None:
        self._llm = llm
        self._web = web
        self._kb = kb
        self._runner = runner
        self._max_steps = max_steps
        names = ["kb", "web", "calc"] + (["python"] if runner else [])
        self._tools = "\n".join(f"- {TOOL_DESCRIPTIONS[n]}" for n in names)

    def run(self, question: str, feedback: str = "") -> Report:
        evidence: list[Evidence] = []
        scratch = ""
        for _ in range(self._max_steps):
            raw = self._llm.complete(
                REACT_PROMPT.format(question=question, feedback=feedback or "ninguna",
                                    tools=self._tools, scratch=scratch)
            ).strip()
            # El modelo puede inventar su propia Observation. Solo el sistema escribe observaciones,
            # así que se descarta todo lo que venga después de la primera Observation del modelo.
            out = raw.split("Observation:", 1)[0].strip()
            scratch += out + "\n"

            action = parse_action(out)
            if action is None:
                final = FINAL_RE.search(out)
                if final:
                    return Report(question, final.group(1).strip(), tuple(evidence))
                scratch += "Observation: formato no válido. Usa Action o Final.\n"
                continue

            found, observation = self._use_tool(*action)
            evidence.extend(found)
            scratch += f"Observation: {observation}\n"

        return Report(question, "", tuple(evidence))

    def _use_tool(self, tool: str, arg: str) -> tuple[list[Evidence], str]:
        """Ejecuta una herramienta. Devuelve la evidencia obtenida y el texto de la Observation."""
        if tool in ("kb", "web"):
            results = self._kb.retrieve(arg) if tool == "kb" else self._web.search(arg)
            return results, format_evidence(results) if results else "sin resultados"
        if tool == "calc":
            try:
                value = calculate(arg)
            except ValueError as e:
                return [], f"error de cálculo: {e}"
            return [Evidence("calc", f"{arg} = {value}")], value
        if self._runner is None:
            return [], "herramienta python no disponible"
        code = extract_code(arg)
        result = self._runner.run(code, CODE_TIMEOUT)
        if not result.ok:
            return [], f"error al ejecutar: {result.output}"
        shown = result.output or "(sin salida)"
        return [Evidence("python", f"código:\n{code}\nsalida: {shown}")], shown
