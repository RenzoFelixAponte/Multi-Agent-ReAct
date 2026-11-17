from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionResult:
    """Resultado de ejecutar código: `ok` es False si falló, agotó el tiempo o excedió los límites."""

    ok: bool
    output: str
