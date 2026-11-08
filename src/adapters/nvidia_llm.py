"""LLM remoto vía NVIDIA API (endpoint compatible con OpenAI). Solo usa la librería estándar.

La key y el modelo se leen de variables de entorno o de un archivo `.env` en la raíz del repo.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from adapters.dotenv_loader import load_dotenv

ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"


def parse_completion(data: dict) -> str:
    """Extrae el texto de la respuesta. Algunos modelos de razonamiento dejan `content` vacío."""
    message = data["choices"][0]["message"]
    return (message.get("content") or "").strip()


class NvidiaLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 max_tokens: int = 1024, timeout: int = 120) -> None:
        load_dotenv()
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self._model = model or os.environ.get("NVIDIA_MODEL") or DEFAULT_MODEL
        self._max_tokens = max_tokens
        self._timeout = timeout
        if not self._api_key:
            raise RuntimeError("Falta NVIDIA_API_KEY: ponla en el archivo .env de la raíz del repo.")

    @property
    def model(self) -> str:
        return self._model

    def complete(self, prompt: str) -> str:
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            ENDPOINT,
            data=body,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"NVIDIA API respondió {e.code}: {detail}") from e
        return parse_completion(data)
