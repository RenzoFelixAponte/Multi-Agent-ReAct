"""Regenera results/tabla_comparativa.csv a partir de results/evaluacion.jsonl, sin GPU ni red.

Sirve para recalcular la tabla cuando cambian las métricas de `eval_summary`, sin volver a correr
los modelos. Solo usa la librería estándar.

Uso:  python scripts/resumir_resultados.py [ruta/evaluacion.jsonl] [ruta/tabla_comparativa.csv]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adapters.jsonl_result_store import JsonlResultStore  # noqa: E402
from application.eval_summary import summarize  # noqa: E402


def main(source: Path, target: Path) -> None:
    table = summarize(JsonlResultStore(source).load())
    if not table:
        raise SystemExit(f"No hay registros en {source}")
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows({k: "" if v is None else v for k, v in row.items()} for row in table)
    print(f"{len(table)} filas -> {target}")


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    main(args[0] if args else ROOT / "results" / "evaluacion.jsonl",
         args[1] if len(args) > 1 else ROOT / "results" / "tabla_comparativa.csv")
