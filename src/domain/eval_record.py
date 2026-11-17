from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalRecord:
    """Resultado de un método sobre una consulta. Se guarda uno por línea para poder reanudar la corrida.

    `faithfulness` y `relevance` van de 0 a 1 (None si el juez no devolvió un formato válido).
    `correct` es la verificación objetiva (GSM8K y MBPP); en WikiQA es None.
    Si `answered` es False el sistema se abstuvo: ambas métricas valen 0.
    """

    dataset: str
    item_id: str
    method: str
    question: str
    answer: str = ""
    answered: bool = False
    faithfulness: float | None = None
    relevance: float | None = None
    correct: bool | None = None
    rounds: int = 1
    evidence_count: int = 0
    seconds: float = 0.0
    error: str = ""
