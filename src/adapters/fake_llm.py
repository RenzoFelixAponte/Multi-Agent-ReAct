"""LLM falso para pruebas: devuelve respuestas predefinidas en orden."""
from __future__ import annotations


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("ScriptedLLM sin respuestas restantes")
        return self._responses.pop(0)
