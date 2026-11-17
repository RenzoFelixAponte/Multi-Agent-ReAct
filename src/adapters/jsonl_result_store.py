"""Guarda un EvalRecord por línea (JSON Lines): cada resultado queda en disco apenas se calcula."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from domain.eval_record import EvalRecord


class JsonlResultStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[EvalRecord]:
        if not self._path.is_file():
            return []
        records: list[EvalRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(EvalRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue  # línea cortada por una interrupción: se recalcula al reanudar
        return records

    def append(self, record: EvalRecord) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            f.flush()
