"""Carga de un archivo .env sin dependencias externas."""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Carga pares CLAVE=VALOR de un .env sin pisar variables que ya existan en el entorno.

    Por defecto busca `.env` en la raíz del repo (dos niveles por encima de src/adapters).
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / ".env"
    env_file = Path(path)
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
