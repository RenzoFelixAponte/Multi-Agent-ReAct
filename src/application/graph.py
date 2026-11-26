"""Grafo de control cíclico (LangGraph) que conecta los agentes mediante puertos."""
from __future__ import annotations

from dataclasses import replace

from langgraph.graph import END, START, StateGraph

from application.auditor import Auditor
from application.flow_state import FlowState
from application.investigator import MAX_REACT_STEPS, Investigator
from application.redactor import Redactor
from ports.code_runner_port import CodeRunnerPort
from ports.knowledge_base_port import KnowledgeBasePort
from ports.llm_port import LLMPort
from ports.web_search_port import WebSearchPort

# Herramientas de cada tarea. En GSM8K y MBPP no se ofrece la web: además de no aportar, podría
# devolver la solución publicada del propio benchmark.
TASK_TOOLS: dict[str, tuple[str, ...]] = {
    "qa": ("kb", "web"),
    "math": ("calc",),
    "code": ("python",),
}
MAX_FEEDBACK_ANSWER_CHARS = 1500


def build_flow(llm: LLMPort, web: WebSearchPort, kb: KnowledgeBasePort, max_rounds: int = 3,
               runner: CodeRunnerPort | None = None, max_steps: int = MAX_REACT_STEPS,
               prefetch: bool = False, task: str | None = None):
    """Construye el grafo: [kb || web] -> investigate -> audit -> (redact | feedback -> investigate | fin).

    Con `prefetch=True`, kb y web se consultan en paralelo (fan-out de LangGraph) antes de que el
    Investigador empiece su bucle ReAct, que luego parte con esa evidencia y puede seguir buscando.
    `task` ("qa", "math" o "code") ajusta las herramientas, el ejemplo del prompt y las reglas del
    Auditor y del Redactor; sin `task`, el Investigador ve todas las herramientas.
    """
    style = task or "general"
    tools = TASK_TOOLS[task] if task else None
    investigator = Investigator(llm, web, kb, max_steps=max_steps, runner=runner, tools=tools, task=style)
    auditor = Auditor(llm, runner=runner, task=style)
    redactor = Redactor(llm, task=style)

    def kb_lookup(state: FlowState) -> dict:
        try:
            return {"prefetched": kb.retrieve(state["question"])}
        except Exception:  # una rama caída no debe tumbar a la otra
            return {"prefetched": []}

    def web_lookup(state: FlowState) -> dict:
        try:
            return {"prefetched": web.search(state["question"])}
        except Exception:
            return {"prefetched": []}

    def investigate(state: FlowState) -> dict:
        report = investigator.run(state["question"], state.get("feedback", ""),
                                  state.get("prefetched", ()))
        return {"report": report, "rounds": state.get("rounds", 0) + 1}

    def audit(state: FlowState) -> dict:
        report = state["report"]
        verdict = auditor.review(report)
        if not verdict.approved:
            return {"verdict": verdict, "rejections": [verdict.reason]}
        if verdict.evidence:  # lo que el Auditor verificó (p. ej. las pruebas ejecutadas) respalda la respuesta
            report = replace(report, evidence=tuple(dict.fromkeys(report.evidence + verdict.evidence)))
        return {"verdict": verdict, "report": report}

    def feedback(state: FlowState) -> dict:
        # Sin su respuesta anterior, el Investigador empieza de cero y suele repetir el mismo error.
        previous = state["report"].answer[:MAX_FEEDBACK_ANSWER_CHARS]
        text = state["verdict"].reason
        if previous:
            text += f"\nTu respuesta rechazada fue:\n{previous}"
        return {"feedback": text}

    def redact(state: FlowState) -> dict:
        return {"final": redactor.write(state["report"])}

    def route(state: FlowState) -> str:
        if state["verdict"].approved:
            return "redact"
        if state.get("rounds", 0) >= max_rounds:
            return "stop"
        return "retry"

    graph = StateGraph(FlowState)
    graph.add_node("investigate", investigate)
    graph.add_node("audit", audit)
    graph.add_node("feedback", feedback)
    graph.add_node("redact", redact)

    if prefetch:
        graph.add_node("kb_lookup", kb_lookup)
        graph.add_node("web_lookup", web_lookup)
        graph.add_edge(START, "kb_lookup")
        graph.add_edge(START, "web_lookup")
        graph.add_edge(["kb_lookup", "web_lookup"], "investigate")  # espera a que terminen ambas
    else:
        graph.add_edge(START, "investigate")
    graph.add_edge("investigate", "audit")
    graph.add_conditional_edges(
        "audit", route, {"redact": "redact", "retry": "feedback", "stop": END}
    )
    graph.add_edge("feedback", "investigate")
    graph.add_edge("redact", END)
    return graph.compile()
