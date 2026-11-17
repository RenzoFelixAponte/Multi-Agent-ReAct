"""Mapeadores puros de filas de Hugging Face a BenchmarkItem. No hacen red."""
from __future__ import annotations

import re
from typing import Any, Callable

from domain.benchmark_item import BenchmarkItem
from domain.evidence import Evidence

_GSM8K_FINAL = re.compile(r"####\s*(.+)$")


def group_wikiqa(rows: list[dict[str, Any]], limit: int) -> list[BenchmarkItem]:
    """WikiQA: una consulta por pregunta que tenga al menos una frase correcta (label=1).

    El contexto conserva TODAS las frases candidatas, también las incorrectas (label=0). Así la base
    vectorial tiene distractores y la recuperación no parte con la respuesta servida en bandeja.
    La referencia es la unión de las frases correctas.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["question_id"]), []).append(row)

    items: list[BenchmarkItem] = []
    for question_id, group in groups.items():
        positives = [r["answer"] for r in group if int(r["label"]) == 1]
        if not positives:
            continue
        items.append(BenchmarkItem(
            dataset="wiki_qa",
            item_id=question_id,
            question=group[0]["question"],
            reference=" ".join(positives),
            context=tuple(Evidence(source=r["document_title"], text=r["answer"]) for r in group),
        ))
        if len(items) >= limit:
            break
    return items


def map_gsm8k(row: dict[str, Any], index: int) -> BenchmarkItem | None:
    """GSM8K: la respuesta final va después de '####'. Sin esa marca la fila se descarta."""
    match = _GSM8K_FINAL.search(row["answer"])
    if match is None:
        return None
    return BenchmarkItem(
        dataset="gsm8k",
        item_id=str(index),
        question=row["question"],
        reference=match.group(1).strip().replace(",", ""),
    )


def map_mbpp(row: dict[str, Any], index: int) -> BenchmarkItem | None:
    """MBPP: el enunciado es 'text', la referencia es el código y `test_list` verifica el código generado."""
    return BenchmarkItem(
        dataset="mbpp",
        item_id=str(row.get("task_id", index)),
        question=row["text"],
        reference=row["code"],
        tests=tuple(row.get("test_list") or ()),
        setup_code=row.get("test_setup_code") or "",
    )


ROW_MAPPERS: dict[str, Callable[[dict[str, Any], int], BenchmarkItem | None]] = {
    "gsm8k": map_gsm8k,
    "mbpp": map_mbpp,
}
DATASET_NAMES = ("wiki_qa", *ROW_MAPPERS)


def map_rows(name: str, rows: list[dict[str, Any]], limit: int) -> list[BenchmarkItem]:
    """Convierte las filas del dataset y descarta las no válidas hasta reunir `limit` items."""
    if name not in DATASET_NAMES:
        raise ValueError(f"dataset desconocido: {name}. Válidos: {sorted(DATASET_NAMES)}")
    if name == "wiki_qa":
        return group_wikiqa(rows, limit)
    mapper = ROW_MAPPERS[name]
    items: list[BenchmarkItem] = []
    for i, row in enumerate(rows):
        item = mapper(row, i)
        if item is not None:
            items.append(item)
        if len(items) >= limit:
            break
    return items


def wikiqa_documents(items: list[BenchmarkItem]) -> list[Evidence]:
    """Documentos para indexar en la base vectorial (frases correctas y distractoras, sin repetir)."""
    seen: set[Evidence] = set()
    documents: list[Evidence] = []
    for item in items:
        for evidence in item.context:
            if evidence not in seen:
                seen.add(evidence)
                documents.append(evidence)
    return documents
