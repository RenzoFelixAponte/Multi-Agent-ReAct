"""Cifras del análisis de errores (docs/analisis_errores.md) a partir de results/evaluacion.jsonl.

Compara los dos métodos consulta por consulta: aciertos discordantes con la prueba exacta de McNemar,
qué pasó con cada rechazo del Auditor y la fidelidad de WikiQA según el largo de la respuesta.
Solo usa la librería estándar y no necesita GPU ni red.

Uso:  python scripts/analisis_errores.py [ruta/evaluacion.jsonl]
"""
from __future__ import annotations

import sys
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adapters.jsonl_result_store import JsonlResultStore  # noqa: E402
from application.eval_summary import latest  # noqa: E402
from application.evaluator import MULTIAGENT, NAIVE, OBJECTIVE_DATASETS  # noqa: E402

SHORT_ANSWER_CHARS = 80


def mcnemar_exact(b: int, c: int) -> float:
    """p-valor bilateral de McNemar exacto: binomial(b + c, 0.5) sobre los pares discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def outcome(record) -> str:
    if not record.answered:
        return "abstención"
    return "correcta" if record.correct else "incorrecta"


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else float("nan")


def main(source: Path) -> None:
    records = {(r.dataset, r.item_id, r.method): r for r in latest(JsonlResultStore(source).load())}
    items = sorted({(d, i) for d, i, _ in records})

    print("## Resultado por consulta (GSM8K y MBPP)\n")
    for dataset in OBJECTIVE_DATASETS:
        pairs = Counter()
        for d, i in items:
            if d == dataset:
                pairs[(outcome(records[(d, i, MULTIAGENT)]), outcome(records[(d, i, NAIVE)]))] += 1
        print(f"{dataset}:")
        for (multi, naive), count in sorted(pairs.items()):
            print(f"  multi-agente {multi:<10} | Naive RAG {naive:<10} -> {count}")

    only_multi = only_naive = 0
    for d, i in items:
        if d in OBJECTIVE_DATASETS:
            multi, naive = records[(d, i, MULTIAGENT)].correct, records[(d, i, NAIVE)].correct
            only_multi += bool(multi and not naive)
            only_naive += bool(naive and not multi)
    print(f"\nDiscordantes: solo acierta el multi-agente {only_multi}, solo el Naive RAG {only_naive}; "
          f"McNemar exacto p = {mcnemar_exact(only_multi, only_naive):.3f}")

    print("\n## Rechazos del Auditor (multi-agente)\n")
    for dataset in ("wiki_qa", *OBJECTIVE_DATASETS):
        group = [records[(d, i, MULTIAGENT)] for d, i in items if d == dataset]
        rejected = [r for r in group if r.rounds > 1]
        recovered = [r for r in rejected if r.answered]
        print(f"{dataset}: {len(rejected)} de {len(group)} con al menos un rechazo; "
              f"{len(recovered)} terminaron con respuesta aprobada "
              f"({sum(bool(r.correct) for r in recovered)} correctas), "
              f"{len(rejected) - len(recovered)} en abstención")
        if dataset in OBJECTIVE_DATASETS:
            approved_wrong = [r for r in group if r.answered and not r.correct]
            print(f"  aprobadas pero incorrectas: {len(approved_wrong)}")

    abstentions = [(d, i) for d, i in items if not records[(d, i, MULTIAGENT)].answered]
    naive_on_them = Counter(outcome(records[(d, i, NAIVE)]) for d, i in abstentions)
    print(f"\nEn las {len(abstentions)} abstenciones del multi-agente, el Naive RAG dio: {dict(naive_on_them)}")

    print("\n## El juez frente a la verificación objetiva (GSM8K y MBPP, ambos métodos)\n")
    answered = [r for r in records.values() if r.dataset in OBJECTIVE_DATASETS and r.answered]
    for label, wanted in (("correctas", True), ("incorrectas", False)):
        group = [r for r in answered if bool(r.correct) == wanted]
        relevance = mean([r.relevance for r in group if r.relevance is not None])
        faithfulness = mean([r.faithfulness for r in group if r.faithfulness is not None])
        print(f"{label}: {len(group)} respuestas, answer relevance {relevance}, faithfulness {faithfulness}")

    print("\n## Fidelidad en WikiQA según el largo de la respuesta\n")
    for method in (MULTIAGENT, NAIVE):
        group = [records[(d, i, method)] for d, i in items if d == "wiki_qa"]
        scored = [r for r in group if r.faithfulness is not None]
        short = [r.faithfulness for r in scored if len(r.answer) < SHORT_ANSWER_CHARS]
        long_ = [r.faithfulness for r in scored if len(r.answer) >= SHORT_ANSWER_CHARS]
        evidence = mean([float(r.evidence_count) for r in group])
        print(f"{method}: cortas (<{SHORT_ANSWER_CHARS} caracteres) {len(short)} con {mean(short)}, "
              f"largas {len(long_)} con {mean(long_)}; fragmentos de contexto en promedio {evidence}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "evaluacion.jsonl")
