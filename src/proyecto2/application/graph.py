"""Grafo de control cíclico (LangGraph) que conecta los agentes mediante puertos."""
from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from proyecto2.application.agents import Auditor, Investigator, Redactor
from proyecto2.domain.models import AuditVerdict, Report
from proyecto2.ports.ports import KnowledgeBasePort, LLMPort, WebSearchPort


class FlowState(TypedDict, total=False):
    question: str
    report: Optional[Report]
    verdict: Optional[AuditVerdict]
    feedback: str
    rounds: int
    final: str


def build_flow(llm: LLMPort, web: WebSearchPort, kb: KnowledgeBasePort, max_rounds: int = 3):
    """Construye el grafo: investigate -> audit -> (redact | feedback -> investigate | fin)."""
    investigator = Investigator(llm, web, kb)
    auditor = Auditor(llm)
    redactor = Redactor(llm)

    def investigate(state: FlowState) -> dict:
        report = investigator.run(state["question"], state.get("feedback", ""))
        return {"report": report, "rounds": state.get("rounds", 0) + 1}

    def audit(state: FlowState) -> dict:
        return {"verdict": auditor.review(state["report"])}

    def feedback(state: FlowState) -> dict:
        return {"feedback": state["verdict"].reason}

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

    graph.set_entry_point("investigate")
    graph.add_edge("investigate", "audit")
    graph.add_conditional_edges(
        "audit", route, {"redact": "redact", "retry": "feedback", "stop": END}
    )
    graph.add_edge("feedback", "investigate")
    graph.add_edge("redact", END)
    return graph.compile()
