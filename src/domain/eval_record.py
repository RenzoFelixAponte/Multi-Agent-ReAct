from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalRecord:
    """Resultado de un método sobre una consulta. Se guarda uno por línea para poder reanudar la corrida.

    `faithfulness` y `relevance` van de 0 a 1 (None si el juez no devolvió un formato válido o falló;
    en ese caso `judge_error` dice por qué).
    `correct` es la verificación objetiva (GSM8K y MBPP); en WikiQA es None.
    Si `answered` es False el sistema se abstuvo: ambas métricas valen 0.
    `evidence` es el contexto que vio el juez, como dicts {source, text}: permite volver a juzgar sin
    volver a generar. `rejections` son los motivos de rechazo del Auditor y `trace` el bucle ReAct de
    la última ronda (vacíos en el Naive RAG). `seconds` mide solo la generación, sin el juez.
    Los registros de la primera corrida no tienen evidencia, traza ni rechazos.
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
    evidence: list[dict[str, str]] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    trace: str = ""
    judge_error: str = ""
