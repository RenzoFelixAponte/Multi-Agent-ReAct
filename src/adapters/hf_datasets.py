"""Adaptador de datasets de Hugging Face (WikiQA, GSM8K, MBPP).

Los mapeadores de filas son funciones puras y se prueban sin red.
`HFDatasetAdapter` es la única parte que descarga datos.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from domain.models import BenchmarkItem, Evidence

# nombre lógico -> (id en Hugging Face, configuración, split por defecto)
DATASET_SPECS: dict[str, tuple[str, str | None]] = {
    "wiki_qa": ("microsoft/wiki_qa", None),
    "gsm8k": ("openai/gsm8k", "main"),
    "mbpp": ("google-research-datasets/mbpp", "full"),
}

_GSM8K_FINAL = re.compile(r"####\s*(.+)$")


def map_wikiqa(row: dict[str, Any], index: int, positives_only: bool = True) -> BenchmarkItem | None:
    """WikiQA: pregunta + frase candidata. Con positives_only solo se conservan las frases correctas (label=1)."""
    if positives_only and int(row["label"]) != 1:
        return None
    evidence = Evidence(source=row["document_title"], text=row["answer"])
    return BenchmarkItem(
        dataset="wiki_qa",
        item_id=str(row.get("question_id", index)),
        question=row["question"],
        reference=row["answer"],
        context=(evidence,),
    )


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
    """MBPP: el enunciado es 'text' y la referencia es el código de la solución."""
    return BenchmarkItem(
        dataset="mbpp",
        item_id=str(row.get("task_id", index)),
        question=row["text"],
        reference=row["code"],
    )


MAPPERS: dict[str, Callable[..., BenchmarkItem | None]] = {
    "wiki_qa": map_wikiqa,
    "gsm8k": map_gsm8k,
    "mbpp": map_mbpp,
}


def map_rows(name: str, rows: list[dict[str, Any]], limit: int) -> list[BenchmarkItem]:
    """Aplica el mapeador del dataset y descarta filas no válidas hasta reunir `limit` items."""
    if name not in MAPPERS:
        raise ValueError(f"dataset desconocido: {name}. Válidos: {sorted(MAPPERS)}")
    mapper = MAPPERS[name]
    items: list[BenchmarkItem] = []
    for i, row in enumerate(rows):
        item = mapper(row, i)
        if item is not None:
            items.append(item)
        if len(items) >= limit:
            break
    return items


class HFDatasetAdapter:
    """Implementa DatasetPort descargando desde Hugging Face con la librería `datasets`."""

    def load(self, name: str, split: str, limit: int) -> list[BenchmarkItem]:
        if name not in DATASET_SPECS:
            raise ValueError(f"dataset desconocido: {name}. Válidos: {sorted(DATASET_SPECS)}")
        from datasets import load_dataset

        hf_id, config = DATASET_SPECS[name]
        ds = load_dataset(hf_id, config, split=split)
        # Se recorre todo el split hasta reunir `limit` items válidos (algunas filas se descartan)
        return map_rows(name, [dict(r) for r in ds], limit)


def wikiqa_documents(items: list[BenchmarkItem]) -> list[Evidence]:
    """Documentos para indexar en la base vectorial a partir del contexto de WikiQA."""
    return [ev for item in items for ev in item.context]
