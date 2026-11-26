"""Corre multi-agente y Naive RAG sobre las mismas consultas y guarda cada resultado apenas se calcula."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any, NamedTuple

from application.answer_checks import numbers_match, passes_tests
from application.eval_set import eval_question
from application.judge import Judge
from application.naive_rag import naive_rag_with_context
from domain.benchmark_item import BenchmarkItem
from domain.eval_record import EvalRecord
from domain.evidence import Evidence
from ports.code_runner_port import CodeRunnerPort
from ports.knowledge_base_port import KnowledgeBasePort
from ports.llm_port import LLMPort
from ports.result_store_port import ResultStorePort

MULTIAGENT = "multiagente"
NAIVE = "naive_rag"
METHODS = (MULTIAGENT, NAIVE)
OBJECTIVE_DATASETS = ("gsm8k", "mbpp")  # los que tienen verificación objetiva de la respuesta


class Outcome(NamedTuple):
    """Lo que produjo un método para una consulta, antes de juzgarlo."""

    answer: str
    evidence: list[Evidence]
    rounds: int = 1
    trace: str = ""
    rejections: tuple[str, ...] = ()


class Evaluator:
    """`flows` asigna a cada dataset su grafo ya construido (con las herramientas que le corresponden)."""

    def __init__(self, flows: Mapping[str, Any], llm: LLMPort, kb: KnowledgeBasePort, judge: Judge,
                 store: ResultStorePort, runner: CodeRunnerPort | None = None) -> None:
        self._flows = flows
        self._llm = llm
        self._kb = kb
        self._judge = judge
        self._store = store
        self._runner = runner

    def run(self, items: list[BenchmarkItem], on_record: Callable[[EvalRecord], None] | None = None) -> None:
        """Salta lo que ya está guardado sin error, así una corrida cortada se reanuda donde quedó."""
        done = {(r.dataset, r.item_id, r.method) for r in self._store.load() if not r.error}
        for item in items:
            for method in METHODS:
                if (item.dataset, item.item_id, method) in done:
                    continue
                record = self._evaluate(item, method)
                self._store.append(record)
                if on_record:
                    on_record(record)

    def _evaluate(self, item: BenchmarkItem, method: str) -> EvalRecord:
        question = eval_question(item)
        start = time.perf_counter()
        base = dict(dataset=item.dataset, item_id=item.item_id, method=method, question=question)
        try:
            out = self._answer(item, question, method)
            seconds = round(time.perf_counter() - start, 2)
            answered = bool(out.answer.strip())
            if answered:
                correct = self._correct(item, out.answer)
            else:  # abstención: el guardrail prefirió no responder
                correct = False if item.dataset in OBJECTIVE_DATASETS else None
        except Exception as e:  # un fallo aislado (red, memoria) no debe perder el resto de la corrida
            return EvalRecord(**base, error=f"{type(e).__name__}: {e}",
                              seconds=round(time.perf_counter() - start, 2))
        record = EvalRecord(**base, answer=out.answer, answered=answered, correct=correct, rounds=out.rounds,
                            evidence_count=len(out.evidence), seconds=seconds,
                            evidence=[{"source": e.source, "text": e.text} for e in out.evidence],
                            rejections=list(out.rejections), trace=out.trace)
        return score(record, self._judge)

    def _answer(self, item: BenchmarkItem, question: str, method: str) -> Outcome:
        if method == NAIVE:
            answer, evidence = naive_rag_with_context(question, self._llm, self._kb)
            return Outcome(answer, evidence)
        state = self._flows[item.dataset].invoke({"question": question})
        report = state.get("report")
        return Outcome(state.get("final", ""), list(report.evidence) if report else [], state.get("rounds", 1),
                       report.trace if report else "", tuple(state.get("rejections", ())))

    def _correct(self, item: BenchmarkItem, answer: str) -> bool | None:
        if item.dataset == "gsm8k":
            return numbers_match(answer, item.reference)
        if item.dataset == "mbpp" and self._runner is not None:
            return passes_tests(answer, item, self._runner)
        return None


def score(record: EvalRecord, judge: Judge) -> EvalRecord:
    """Pone las métricas del juez. Una abstención puntúa 0 en ambas.

    Si el juez falla (red, cuota), la métrica queda en None y el error en `judge_error`: la respuesta
    ya generada no se pierde y se puede volver a juzgar con `rejudge`.
    """
    if not record.answered:
        return replace(record, faithfulness=0.0, relevance=0.0, judge_error="")
    evidence = [Evidence(e["source"], e["text"]) for e in record.evidence]
    errors: list[str] = []

    def safe(metric, *args):
        try:
            return metric(*args)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
            return None

    return replace(record,
                   faithfulness=safe(judge.faithfulness, record.answer, evidence, record.question),
                   relevance=safe(judge.relevance, record.question, record.answer),
                   judge_error="; ".join(errors))


def rejudge(records: Iterable[EvalRecord], judge: Judge, only_missing: bool = True) -> Iterable[EvalRecord]:
    """Vuelve a juzgar respuestas ya generadas con la evidencia guardada, sin volver a generarlas.

    Con `only_missing`, solo las que quedaron sin puntaje. Se saltan los registros con error y los que
    tenían evidencia pero no la guardaron (los de la primera corrida). Al agregar los nuevos registros
    al mismo archivo, `latest` se queda con ellos.
    """
    for record in records:
        if record.error or (record.evidence_count and not record.evidence):
            continue
        if only_missing and record.faithfulness is not None and record.relevance is not None:
            continue
        yield score(record, judge)
