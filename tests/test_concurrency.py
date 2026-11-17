import threading

from adapters.fake_llm import ScriptedLLM
from adapters.fake_search import StaticSearch
from adapters.in_memory_kb import InMemoryKB
from application.graph import build_flow
from domain.evidence import Evidence


class _BarrierKB:
    """Cada consulta espera a que la otra rama también esté consultando: solo avanza si corren a la vez."""

    def __init__(self, barrier: threading.Barrier) -> None:
        self._barrier = barrier

    def retrieve(self, query: str, k: int = 3) -> list[Evidence]:
        self._barrier.wait(timeout=3)
        return [Evidence("kb", "Lima es la capital del Perú.")]


class _BarrierWeb:
    def __init__(self, barrier: threading.Barrier) -> None:
        self._barrier = barrier

    def search(self, query: str, k: int = 3) -> list[Evidence]:
        self._barrier.wait(timeout=3)
        return [Evidence("web", "Lima es la capital del Perú, según la web.")]


def test_prefetch_queries_kb_and_web_at_the_same_time_and_merges_evidence():
    barrier = threading.Barrier(2)
    llm = ScriptedLLM(["Final: Lima", "APROBADO", "Lima [kb][web]."])
    flow = build_flow(llm, _BarrierWeb(barrier), _BarrierKB(barrier), prefetch=True)

    state = flow.invoke({"question": "¿Capital del Perú?"})

    assert sorted(e.source for e in state["prefetched"]) == ["kb", "web"]
    assert sorted(e.source for e in state["report"].evidence) == ["kb", "web"]
    assert state["verdict"].approved
    # El investigador ya vio la evidencia paralela en su primer prompt
    assert "recuperación inicial" in llm.prompts[0]
    assert "Lima es la capital del Perú, según la web." in llm.prompts[0]


def test_prefetch_survives_a_failing_branch():
    class _BrokenWeb:
        def search(self, query: str, k: int = 3):
            raise ConnectionError("sin red")

    llm = ScriptedLLM(["Final: Lima", "APROBADO", "Lima."])
    docs = [Evidence("kb", "Lima es la capital del Perú.")]
    flow = build_flow(llm, _BrokenWeb(), InMemoryKB(docs), prefetch=True)
    state = flow.invoke({"question": "capital del Perú"})
    assert [e.source for e in state["report"].evidence] == ["kb"]
    assert state["verdict"].approved


def test_tool_failure_becomes_an_observation_not_a_crash():
    class _BrokenWeb:
        def search(self, query: str, k: int = 3):
            raise ConnectionError("sin red")

    llm = ScriptedLLM(["Action: web[algo]", "Final: no pude buscar"])
    flow = build_flow(llm, _BrokenWeb(), InMemoryKB([]), max_rounds=1)
    state = flow.invoke({"question": "q"})
    assert "error de la herramienta web: sin red" in llm.prompts[1]
    assert not state["verdict"].approved


def test_without_prefetch_graph_has_no_parallel_nodes():
    flow = build_flow(ScriptedLLM([]), StaticSearch([]), InMemoryKB([]))
    assert "kb_lookup" not in flow.get_graph().nodes
    flow = build_flow(ScriptedLLM([]), StaticSearch([]), InMemoryKB([]), prefetch=True)
    assert {"kb_lookup", "web_lookup"} <= set(flow.get_graph().nodes)
