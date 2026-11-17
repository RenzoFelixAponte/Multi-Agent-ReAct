import json

import pytest

from adapters.fake_llm import ScriptedLLM
from adapters.in_memory_kb import InMemoryKB
from adapters.jsonl_result_store import JsonlResultStore
from adapters.subprocess_code_runner import SubprocessCodeRunner
from application.eval_set import build_eval_set, eval_question
from application.eval_summary import summarize
from application.evaluator import MULTIAGENT, NAIVE, Evaluator
from application.judge import Judge, parse_faithfulness, parse_relevance
from domain.benchmark_item import BenchmarkItem
from domain.audit_verdict import AuditVerdict
from domain.eval_record import EvalRecord
from domain.evidence import Evidence
from domain.report import Report

DOCS = [Evidence("wiki", "Lima es la capital del Perú."), Evidence("wiki", "Janet tiene 16 huevos.")]
WIKI = BenchmarkItem("wiki_qa", "Q1", "¿Capital del Perú?", "Lima es la capital del Perú.")
GSM = BenchmarkItem("gsm8k", "0", "Janet tiene 16 huevos...", "18")
MBPP = BenchmarkItem("mbpp", "11", "Suma dos números", "", tests=("assert add(1, 2) == 3",))


class FakeFlow:
    """Grafo falso: devuelve un estado fijo por pregunta y cuenta las invocaciones."""

    def __init__(self, final: str = "", evidence=(Evidence("calc", "x = 1"),), rounds: int = 1) -> None:
        self.final, self.evidence, self.rounds, self.calls = final, tuple(evidence), rounds, 0

    def invoke(self, state: dict) -> dict:
        self.calls += 1
        out = {"report": Report(state["question"], self.final, self.evidence), "rounds": self.rounds,
               "verdict": AuditVerdict(bool(self.final))}
        if self.final:
            out["final"] = self.final
        return out


def test_parse_faithfulness_counts_supported_claims():
    assert parse_faithfulness("SI: a\nNO: b\n- SÍ: c\n2) no: d") == 0.5
    assert parse_faithfulness("SI: única") == 1.0
    # formato real de Mistral: numerado, con el veredicto entre paréntesis y en inglés
    assert parse_faithfulness("1. Janet tiene 16 huevos. (NO) - no está.\n2. Le quedan 9. (YES) - se infiere.") == 0.5
    assert parse_faithfulness("no pude evaluar") is None


def test_parse_relevance_normalizes_and_rejects_out_of_range():
    assert parse_relevance("PUNTAJE: 8") == 0.8
    assert parse_relevance("Puntaje: 10") == 1.0
    assert parse_relevance("sin número") is None
    assert parse_relevance("PUNTAJE: 11") is None


def test_judge_retries_once_when_format_is_invalid_and_gives_up_after():
    llm = ScriptedLLM(["no sé", "SI: ok\nSI: ok2"])
    assert Judge(llm).faithfulness("resp", DOCS) == 1.0
    assert len(llm.prompts) == 2

    llm = ScriptedLLM(["???", "???"])
    assert Judge(llm).relevance("q", "a") is None


def test_judge_gives_zero_faithfulness_without_calling_llm_when_there_is_no_context():
    llm = ScriptedLLM([])
    assert Judge(llm).faithfulness("algo", []) == 0.0
    assert llm.prompts == []


def test_eval_question_includes_mbpp_tests_and_build_eval_set_mixes_datasets():
    assert "assert add(1, 2) == 3" in eval_question(MBPP)
    assert eval_question(WIKI) == WIKI.question

    class Fake:
        def load(self, name, split, limit):
            return [BenchmarkItem(name, str(i), "q", "r") for i in range(limit)]

    items = build_eval_set(Fake())
    assert len(items) == 50
    assert [sum(i.dataset == d for i in items) for d in ("wiki_qa", "gsm8k", "mbpp")] == [20, 15, 15]


def _evaluator(tmp_path, flows, llm, runner=None):
    store = JsonlResultStore(tmp_path / "r.jsonl")
    return Evaluator(flows, llm, InMemoryKB(DOCS), Judge(llm), store, runner), store


def test_evaluator_scores_both_methods_with_objective_check(tmp_path):
    flow = FakeFlow(final="Respuesta final: 18")
    llm = ScriptedLLM([
        "SI: a", "PUNTAJE: 9",                      # juez sobre el multi-agente
        "Respuesta final: 17", "SI: a\nNO: b", "PUNTAJE: 5",   # naive: respuesta + juez
    ])
    evaluator, store = _evaluator(tmp_path, {"gsm8k": flow}, llm)
    evaluator.run([GSM])

    by_method = {r.method: r for r in store.load()}
    assert by_method[MULTIAGENT].correct is True
    assert by_method[MULTIAGENT].faithfulness == 1.0 and by_method[MULTIAGENT].relevance == 0.9
    assert by_method[NAIVE].correct is False
    assert by_method[NAIVE].faithfulness == 0.5 and by_method[NAIVE].relevance == 0.5


def test_abstention_scores_zero_without_calling_the_judge(tmp_path):
    flow = FakeFlow(final="")
    llm = ScriptedLLM(["Lima", "SI: a", "PUNTAJE: 10"])   # solo naive usa el LLM
    evaluator, store = _evaluator(tmp_path, {"wiki_qa": flow}, llm)
    evaluator.run([WIKI])

    multi = next(r for r in store.load() if r.method == MULTIAGENT)
    assert (multi.answered, multi.faithfulness, multi.relevance, multi.correct) == (False, 0.0, 0.0, None)


def test_mbpp_answer_is_checked_by_running_the_tests(tmp_path):
    flow = FakeFlow(final="```python\ndef add(a, b):\n    return a + b\n```")
    llm = ScriptedLLM(["SI: a", "PUNTAJE: 10", "def add(a, b): return a - b", "SI: a", "PUNTAJE: 10"])
    evaluator, store = _evaluator(tmp_path, {"mbpp": flow}, llm, SubprocessCodeRunner())
    evaluator.run([MBPP])
    by_method = {r.method: r for r in store.load()}
    assert by_method[MULTIAGENT].correct is True
    assert by_method[NAIVE].correct is False


def test_run_resumes_skipping_finished_items_and_retries_errors(tmp_path):
    flow = FakeFlow(final="Lima")
    llm = ScriptedLLM(["SI: a", "PUNTAJE: 10", "Lima", "SI: a", "PUNTAJE: 10"])
    evaluator, store = _evaluator(tmp_path, {"wiki_qa": flow}, llm)
    evaluator.run([WIKI])
    assert flow.calls == 1

    evaluator.run([WIKI])  # segunda corrida: ya está todo guardado
    assert flow.calls == 1 and len(store.load()) == 2

    class Boom:
        def invoke(self, state):
            raise RuntimeError("memoria agotada")

    store2 = JsonlResultStore(tmp_path / "r2.jsonl")
    llm2 = ScriptedLLM(["Lima", "SI: a", "PUNTAJE: 10"])
    failing = Evaluator({"wiki_qa": Boom()}, llm2, InMemoryKB(DOCS), Judge(llm2), store2)
    failing.run([WIKI])
    errors = [r for r in store2.load() if r.error]
    assert len(errors) == 1 and "memoria agotada" in errors[0].error

    llm3 = ScriptedLLM(["SI: a", "PUNTAJE: 10"])
    retry = Evaluator({"wiki_qa": FakeFlow(final="Lima")}, llm3, InMemoryKB(DOCS), Judge(llm3), store2)
    retry.run([WIKI])  # solo reintenta el que falló
    multi = [r for r in summarize(store2.load()) if r["método"] == MULTIAGENT and r["dataset"] == "todos"][0]
    assert multi["n"] == 1 and multi["errores"] == 0


def test_store_ignores_truncated_last_line(tmp_path):
    path = tmp_path / "r.jsonl"
    store = JsonlResultStore(path)
    store.append(EvalRecord("wiki_qa", "Q1", NAIVE, "q"))
    with path.open("a", encoding="utf-8") as f:
        f.write('{"dataset": "wiki_qa", "item_id": "Q2", "meth')   # corrida interrumpida a la mitad
    assert [r.item_id for r in store.load()] == ["Q1"]
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["item_id"] == "Q1"


def test_summarize_groups_by_method_and_dataset_with_overall_row():
    records = [
        EvalRecord("gsm8k", "0", MULTIAGENT, "q", "a", True, 1.0, 0.8, True, 2, 3, 10.0),
        EvalRecord("gsm8k", "1", MULTIAGENT, "q", "", False, 0.0, 0.0, False, 3, 0, 20.0),
        EvalRecord("wiki_qa", "Q", MULTIAGENT, "q", "a", True, 0.5, 1.0, None, 1, 2, 6.0),
        EvalRecord("wiki_qa", "Q", NAIVE, "q", "a", True, 0.25, 0.5, None, 1, 3, 2.0),
    ]
    table = {(r["método"], r["dataset"]): r for r in summarize(records)}
    gsm = table[(MULTIAGENT, "gsm8k")]
    assert (gsm["n"], gsm["respondidas"], gsm["faithfulness"], gsm["exactitud"]) == (2, 0.5, 0.5, 0.5)
    assert gsm["faithfulness_respondidas"] == 1.0  # la abstención no entra en este promedio
    everything = table[(MULTIAGENT, "todos")]
    assert everything["n"] == 3 and everything["faithfulness"] == pytest.approx(0.5)
    assert table[(NAIVE, "todos")]["exactitud"] is None
