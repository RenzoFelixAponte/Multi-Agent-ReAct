"""Corre multi-agente y Naive RAG sobre las mismas consultas y guarda cada resultado apenas se calcula."""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

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
            answer, evidence, rounds = self._answer(item, question, method)
            answered = bool(answer.strip())
            if answered:
                faithfulness = self._judge.faithfulness(answer, evidence)
                relevance = self._judge.relevance(question, answer)
                correct = self._correct(item, answer)
            else:  # abstención: el guardrail prefirió no responder, y eso puntúa 0
                faithfulness = relevance = 0.0
                correct = False if item.dataset in OBJECTIVE_DATASETS else None
            return EvalRecord(**base, answer=answer, answered=answered, faithfulness=faithfulness,
                              relevance=relevance, correct=correct, rounds=rounds,
                              evidence_count=len(evidence), seconds=round(time.perf_counter() - start, 2))
        except Exception as e:  # un fallo aislado (red, memoria) no debe perder el resto de la corrida
            return EvalRecord(**base, error=f"{type(e).__name__}: {e}",
                              seconds=round(time.perf_counter() - start, 2))

    def _answer(self, item: BenchmarkItem, question: str, method: str) -> tuple[str, list[Evidence], int]:
        if method == NAIVE:
            answer, evidence = naive_rag_with_context(question, self._llm, self._kb)
            return answer, evidence, 1
        state = self._flows[item.dataset].invoke({"question": question})
        report = state.get("report")
        return state.get("final", ""), list(report.evidence) if report else [], state.get("rounds", 1)

    def _correct(self, item: BenchmarkItem, answer: str) -> bool | None:
        if item.dataset == "gsm8k":
            return numbers_match(answer, item.reference)
        if item.dataset == "mbpp" and self._runner is not None:
            return passes_tests(answer, item, self._runner)
        return None
