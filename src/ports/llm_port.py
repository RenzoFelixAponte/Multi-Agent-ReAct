from __future__ import annotations

from typing import Protocol


class LLMPort(Protocol):
    """Contrato para cualquier modelo de lenguaje (local o remoto)."""

    def complete(self, prompt: str) -> str: ...
