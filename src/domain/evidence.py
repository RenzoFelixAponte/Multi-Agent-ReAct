from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    """Fragmento de contexto recuperado (de la base vectorial o de la web)."""

    source: str
    text: str
