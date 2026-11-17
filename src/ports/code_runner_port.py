from __future__ import annotations

from typing import Protocol

from domain.execution_result import ExecutionResult


class CodeRunnerPort(Protocol):
    """Contrato para ejecutar código Python de forma aislada y con tiempo límite."""

    def run(self, code: str, timeout: float = 5.0) -> ExecutionResult: ...
