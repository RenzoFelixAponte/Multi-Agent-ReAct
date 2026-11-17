from __future__ import annotations

import re
from collections.abc import Sequence

from application.answer_checks import extract_code, has_code_block, last_executed_function
from application.calculator import calculate
from application.formatting import format_evidence
from application.prompts import LAST_STEP_NOTICE, REACT_EXAMPLES, REACT_PROMPT, TOOL_DESCRIPTIONS
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
    """Bucle ReAct: Thought -> Action -> Observation hasta obtener Final.

    `tools` limita las herramientas que ve y puede usar (por defecto kb, web, calc y, si hay
    `runner`, python). `task` elige el ejemplo resuelto del prompt ("qa", "math" o "code").
    """

    def __init__(self, llm: LLMPort, web: WebSearchPort, kb: KnowledgeBasePort,
                 max_steps: int = MAX_REACT_STEPS, runner: CodeRunnerPort | None = None,
                 tools: Sequence[str] | None = None, task: str = "general") -> None:
        self._llm = llm
        self._web = web
        self._kb = kb
        self._runner = runner
        self._max_steps = max_steps
        names = list(tools) if tools is not None else ["kb", "web", "calc"] + (["python"] if runner else [])
        if runner is None and "python" in names:
            names.remove("python")
        self._names = frozenset(names)
        self._tools = "\n".join(f"- {TOOL_DESCRIPTIONS[n]}" for n in names)
        self._example = REACT_EXAMPLES[task]

    def run(self, question: str, feedback: str = "", prefetched: Sequence[Evidence] = ()) -> Report:
        evidence: list[Evidence] = list(prefetched)
        scratch = ""
        if prefetched:
            scratch = ("Observation (recuperación inicial de kb y web, hecha en paralelo): "
                       f"{format_evidence(prefetched)}\n")
        done: set[tuple[str, str]] = set()
        for step in range(self._max_steps):
            last = LAST_STEP_NOTICE if step == self._max_steps - 1 else ""
            raw = self._llm.complete(
                REACT_PROMPT.format(question=question, feedback=feedback or "ninguna", tools=self._tools,
                                    example=self._example, scratch=scratch, last_step=last)
            ).strip()
            # El modelo puede inventar su propia Observation. Solo el sistema escribe observaciones,
            # así que se descarta todo lo que venga después de la primera Observation del modelo.
            out = raw.split("Observation:", 1)[0].strip()
            scratch += out + "\n"

            action = parse_action(out)
            if action is None:
                final = FINAL_RE.search(out)
                if final:
                    return Report(question, self._deliverable(final.group(1).strip(), evidence),
                                  tuple(dict.fromkeys(evidence)))
                scratch += "Observation: formato no válido. Usa Action o Final.\n"
                continue

            if action in done:  # los 7B suelen repetir la misma Action en bucle en vez de responder
                scratch += ("Observation: ya ejecutaste esa Action y tienes su resultado arriba. "
                            "No la repitas: escribe Final con la información disponible.\n")
                continue
            done.add(action)
            found, observation = self._use_tool(*action)
            if len(ACTION_START_RE.findall(out)) > 1:
                observation += "\n(Solo se ejecutó la primera Action. Escribe una sola Action por turno.)"
            evidence.extend(found)
            scratch += f"Observation: {observation}\n"

        return Report(question, "", tuple(dict.fromkeys(evidence)))

    def _deliverable(self, answer: str, evidence: list[Evidence]) -> str:
        """Si el modelo dice que su código funciona pero no lo copia en Final, se entrega el código
        que realmente ejecutó: transcribirlo de nuevo es justo donde un 7B introduce errores."""
        if "python" in self._names and not has_code_block(answer):
            code = last_executed_function(evidence)
            if code:
                return f"{answer}\n```python\n{code}\n```"
        return answer

    def _use_tool(self, tool: str, arg: str) -> tuple[list[Evidence], str]:
        """Ejecuta una herramienta. Devuelve la evidencia obtenida y el texto de la Observation."""
        if tool not in self._names:
            return [], f"herramienta {tool} no disponible; usa solo las herramientas listadas"
        if tool in ("kb", "web"):
            try:
                results = self._kb.retrieve(arg) if tool == "kb" else self._web.search(arg)
            except Exception as e:  # una búsqueda caída no debe tumbar el flujo
                return [], f"error de la herramienta {tool}: {e}"
            return results, format_evidence(results) if results else "sin resultados"
        if tool == "calc":
            try:
                value = calculate(arg)
            except ValueError as e:
                return [], f"error de cálculo: {e}"
            return [Evidence("calc", f"{arg} = {value}")], value
        code = extract_code(arg)
        result = self._runner.run(code, CODE_TIMEOUT)
        if not result.ok:
            return [], f"error al ejecutar: {result.output}"
        shown = result.output or "(sin salida)"
        return [Evidence("python", f"código:\n{code}\nsalida: {shown}")], shown
