"""Adaptador que descarga datasets de Hugging Face y los convierte en BenchmarkItem."""
from __future__ import annotations

from adapters.hf_mappers import MAPPERS, map_rows
from domain.benchmark_item import BenchmarkItem

# nombre lógico -> (id en Hugging Face, configuración)
DATASET_SPECS: dict[str, tuple[str, str | None]] = {
    "wiki_qa": ("microsoft/wiki_qa", None),
    "gsm8k": ("openai/gsm8k", "main"),
    "mbpp": ("google-research-datasets/mbpp", "full"),
}


class HFDatasetAdapter:
    """Implementa DatasetPort descargando desde Hugging Face con la librería `datasets`."""

    def load(self, name: str, split: str, limit: int) -> list[BenchmarkItem]:
        if name not in DATASET_SPECS or name not in MAPPERS:
            raise ValueError(f"dataset desconocido: {name}. Válidos: {sorted(DATASET_SPECS)}")
        from datasets import load_dataset

        hf_id, config = DATASET_SPECS[name]
        ds = load_dataset(hf_id, config, split=split)
        # Se recorre todo el split hasta reunir `limit` items válidos (algunas filas se descartan)
        return map_rows(name, [dict(r) for r in ds], limit)
