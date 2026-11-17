from __future__ import annotations

from typing import Protocol

from domain.eval_record import EvalRecord


class ResultStorePort(Protocol):
    """Contrato para guardar resultados parciales y recuperarlos al reanudar."""

    def load(self) -> list[EvalRecord]: ...

    def append(self, record: EvalRecord) -> None: ...
