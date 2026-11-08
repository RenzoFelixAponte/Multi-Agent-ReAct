from adapters.fake_llm import ScriptedLLM
from adapters.fake_search import StaticSearch
from adapters.in_memory_kb import InMemoryKB
from application.graph import build_flow
from application.naive_rag import naive_rag
from domain.evidence import Evidence

DOCS = [Evidence("wiki", "Lima es la capital del Perú.")]


def test_flow_rejects_then_approves_and_redacts():
    llm = ScriptedLLM([
        "Action: kb[capital Perú]",                 # investigador, paso 1
        "Final: Lima",                              # investigador, paso 2
        "RECHAZADO: falta fuente web",              # auditor, ronda 1
        "Action: web[Lima capital Perú]",           # investigador, ronda 2
        "Final: Lima es la capital del Perú",       # investigador, ronda 2
        "APROBADO",                                 # auditor, ronda 2
        "Lima es la capital del Perú [wiki].",      # redactor
    ])
    flow = build_flow(llm, StaticSearch([Evidence("web", "Lima es la capital.")]), InMemoryKB(DOCS))
    state = flow.invoke({"question": "¿Cuál es la capital del Perú?"})

    assert state["verdict"].approved
    assert state["rounds"] == 2
    assert state["final"] == "Lima es la capital del Perú [wiki]."
    # El feedback del auditor llega al investigador en la segunda ronda
    assert "falta fuente web" in llm.prompts[3]


def test_flow_stops_after_max_rounds_without_redacting():
    llm = ScriptedLLM([
        "Final: no sé",                              # sin evidencia -> rechazo sin llamar al LLM del auditor
        "Final: no sé",
    ])
    flow = build_flow(llm, StaticSearch([]), InMemoryKB([]), max_rounds=2)
    state = flow.invoke({"question": "pregunta sin contexto"})

    assert not state["verdict"].approved
    assert state["rounds"] == 2
    assert "final" not in state


def test_naive_rag_is_single_call():
    llm = ScriptedLLM(["Lima"])
    answer = naive_rag("capital Perú", llm, InMemoryKB(DOCS))
    assert answer == "Lima"
    assert len(llm.prompts) == 1
