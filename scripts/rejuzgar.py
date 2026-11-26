"""Vuelve a juzgar una corrida con la evidencia guardada, sin volver a generar las respuestas.

Por defecto solo juzga lo que quedó sin puntaje (por ejemplo, si la API del juez falló). Con
--todos juzga todo de nuevo, por ejemplo para comparar otro juez sobre las mismas respuestas.
El juez es NvidiaLLM (NVIDIA_API_KEY y, opcional, NVIDIA_MODEL en el .env). Los registros nuevos se
agregan al mismo archivo y la tabla se queda con el último de cada consulta.

Uso:  python scripts/rejuzgar.py results/<corrida>/evaluacion.jsonl [--todos]
      python scripts/resumir_resultados.py results/<corrida>/evaluacion.jsonl results/<corrida>/tabla_comparativa.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adapters.jsonl_result_store import JsonlResultStore  # noqa: E402
from adapters.nvidia_llm import NvidiaLLM  # noqa: E402
from application.eval_summary import latest  # noqa: E402
from application.evaluator import rejudge  # noqa: E402
from application.judge import Judge  # noqa: E402


def main(path: Path, everything: bool) -> None:
    store = JsonlResultStore(path)
    records = latest(store.load())
    judge = Judge(NvidiaLLM())
    count = 0
    for record in rejudge(records, judge, only_missing=not everything):
        store.append(record)  # uno por uno: si se corta, lo juzgado ya quedó guardado
        count += 1
        print(f"[{record.dataset:7}] {record.item_id:>6} {record.method:12} "
              f"faith={record.faithfulness} rel={record.relevance} {record.judge_error}")
    print(f"{count} registros juzgados de nuevo en {path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--todos"]
    if len(args) != 1:
        raise SystemExit(__doc__)
    main(Path(args[0]), "--todos" in sys.argv[1:])
