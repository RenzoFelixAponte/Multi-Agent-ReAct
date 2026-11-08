from __future__ import annotations

from typing import Protocol

from domain.benchmark_item import BenchmarkItem


class DatasetPort(Protocol):
    """Contrato para cargar datasets de evaluación."""

    def load(self, name: str, split: str, limit: int) -> list[BenchmarkItem]: ...
