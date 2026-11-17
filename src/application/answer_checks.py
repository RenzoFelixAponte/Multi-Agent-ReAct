"""Verificación objetiva de respuestas: número final de GSM8K y pruebas unitarias de MBPP."""
from __future__ import annotations

import re
from collections.abc import Sequence

from domain.benchmark_item import BenchmarkItem
from domain.evidence import Evidence
from ports.code_runner_port import CodeRunnerPort

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_FINAL_LINE = re.compile(r"(?:respuesta\s+final|final\s+answer|respuesta|answer)\s*[:=]\s*(.+)", re.I)
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S | re.I)


def _numbers(text: str) -> list[str]:
    return _NUMBER.findall(re.sub(r"(?<=\d),(?=\d{3})", "", text))


def extract_final_number(text: str) -> float | None:
    """Número tras 'Respuesta final:' si existe; si no, el último número del texto."""
    for line in reversed(text.strip().splitlines()):
        match = _FINAL_LINE.search(line)
        if match and _numbers(match.group(1)):
            return float(_numbers(match.group(1))[0])
    found = _numbers(text)
    return float(found[-1]) if found else None


def numbers_match(answer: str, reference: str, tolerance: float = 1e-4) -> bool:
    expected = _numbers(reference)
    got = extract_final_number(answer)
    if got is None or not expected:
        return False
    return abs(got - float(expected[-1])) <= tolerance


def has_code_block(text: str) -> bool:
    return _FENCE.search(text) is not None


def extract_code(text: str) -> str:
    """Primer bloque ```python``` del texto; si no hay bloques, el texto completo."""
    match = _FENCE.search(text)
    return (match.group(1) if match else text).strip()


def passes_tests(answer: str, item: BenchmarkItem, runner: CodeRunnerPort, timeout: float = 10.0) -> bool:
    """Ejecuta el código de la respuesta junto con los `assert` de MBPP."""
    if not item.tests:
        return False
    program = "\n".join([item.setup_code, extract_code(answer), *item.tests])
    return runner.run(program, timeout).ok


def code_was_executed(answer: str, evidence: Sequence[Evidence]) -> bool:
    """El primer bloque de código de la respuesta aparece tal cual en una ejecución exitosa de `python`."""
    squash = lambda t: re.sub(r"\s+", "", t)
    code = squash(extract_code(answer))
    return any(e.source == "python" and code in squash(e.text) for e in evidence)

