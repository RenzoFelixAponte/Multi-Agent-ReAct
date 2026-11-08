"""Puertos: contratos que el núcleo necesita del mundo exterior.

Los adaptadores (Mistral local, ChromaDB, DuckDuckGo, etc.) implementan estos
Protocol. El núcleo solo conoce estas interfaces.
"""
from __future__ import annotations

from typing import Protocol

from proyecto2.domain.models import BenchmarkItem, Evidence


class LLMPort(Protocol):
    def complete(self, prompt: str) -> str: ...


class WebSearchPort(Protocol):
    def search(self, query: str, k: int = 3) -> list[Evidence]: ...


class KnowledgeBasePort(Protocol):
    def retrieve(self, query: str, k: int = 3) -> list[Evidence]: ...


class DatasetPort(Protocol):
    def load(self, name: str, split: str, limit: int) -> list[BenchmarkItem]: ...
