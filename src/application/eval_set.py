"""Arma el set de consultas de evaluación mezclando WikiQA, GSM8K y MBPP."""
from __future__ import annotations

from domain.benchmark_item import BenchmarkItem
from ports.dataset_port import DatasetPort

# (dataset, split, cantidad): 20 + 15 + 15 = 50 consultas
EVAL_MIX: tuple[tuple[str, str, int], ...] = (
    ("wiki_qa", "validation", 20),
    ("gsm8k", "test", 15),
    ("mbpp", "test", 15),
)


def build_eval_set(datasets: DatasetPort, mix: tuple[tuple[str, str, int], ...] = EVAL_MIX) -> list[BenchmarkItem]:
    return [item for name, split, count in mix for item in datasets.load(name, split, count)]


def eval_question(item: BenchmarkItem) -> str:
    """Pregunta que reciben los sistemas. En MBPP incluye las pruebas, como en Austin et al. (2021),
    para que el modelo conozca el nombre y la firma de la función."""
    if item.tests:
        return f"{item.question}\nTu código debe pasar estas pruebas:\n" + "\n".join(item.tests)
    return item.question
