"""Tabla comparativa: promedia las métricas por método y por dataset."""
from __future__ import annotations

from collections.abc import Iterable

from domain.eval_record import EvalRecord


def latest(records: Iterable[EvalRecord]) -> list[EvalRecord]:
    """Si una consulta se reintentó, queda solo el último registro de cada (dataset, item, método)."""
    by_key = {(r.dataset, r.item_id, r.method): r for r in records}
    return list(by_key.values())


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _row(method: str, dataset: str, group: list[EvalRecord]) -> dict:
    valid = [r for r in group if not r.error]
    with_truth = [r for r in valid if r.correct is not None]
    return {
        "método": method,
        "dataset": dataset,
        "n": len(group),
        "respondidas": _mean([float(r.answered) for r in valid]),
        "faithfulness": _mean([r.faithfulness for r in valid if r.faithfulness is not None]),
        "faithfulness_respondidas": _mean([r.faithfulness for r in valid
                                           if r.answered and r.faithfulness is not None]),
        "answer_relevance": _mean([r.relevance for r in valid if r.relevance is not None]),
        "exactitud": _mean([float(r.correct) for r in with_truth]),
        "rondas": _mean([float(r.rounds) for r in valid]),
        "seg_por_consulta": _mean([r.seconds for r in valid]),
        "errores": len(group) - len(valid),
        "juez_fallido": sum(r.faithfulness is None or r.relevance is None for r in valid),
    }


def summarize(records: Iterable[EvalRecord]) -> list[dict]:
    """Una fila por (método, dataset) y una fila 'todos' por método.

    Las abstenciones puntúan 0 en faithfulness y relevance (`respondidas` muestra cuántas fueron).
    `faithfulness_respondidas` promedia solo las respuestas entregadas: separa "qué tan fiel es
    cuando responde" de "cuántas veces se abstiene".
    `exactitud` solo cuenta GSM8K (número final) y MBPP (pruebas unitarias), que tienen verificación objetiva.
    """
    rows = latest(records)
    table: list[dict] = []
    for method in sorted({r.method for r in rows}):
        of_method = [r for r in rows if r.method == method]
        for dataset in sorted({r.dataset for r in of_method}):
            table.append(_row(method, dataset, [r for r in of_method if r.dataset == dataset]))
        table.append(_row(method, "todos", of_method))
    return table
