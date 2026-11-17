"""Ejecuta código Python en un proceso aparte con tiempo límite y límites de memoria y CPU.

Aislamiento básico: intérprete en modo aislado (`-I`), directorio temporal propio, entorno vacío y
`setrlimit` dentro del hijo. No bloquea la red ni la lectura de archivos del sistema, así que está
pensado para correrse en una VM desechable como la de Colab.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from domain.execution_result import ExecutionResult

_BOOTSTRAP = """\
import sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, ({memory}, {memory}))
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
except (ImportError, ValueError, OSError):
    pass
path = sys.argv[1]
sys.argv = [path]
exec(compile(open(path, encoding="utf-8").read(), path, "exec"), {{"__name__": "__main__"}})
"""


class SubprocessCodeRunner:
    def __init__(self, memory_mb: int = 1024, max_output: int = 2000) -> None:
        self._memory = memory_mb * 1024 * 1024
        self._max_output = max_output

    def run(self, code: str, timeout: float = 5.0) -> ExecutionResult:
        cpu = int(timeout) + 1
        bootstrap = _BOOTSTRAP.format(memory=self._memory, cpu=cpu)
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "main.py"
            script.write_text(code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-c", bootstrap, str(script)],
                    cwd=tmp, capture_output=True, text=True, timeout=timeout, env={},
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(False, f"tiempo límite excedido ({timeout:g} s)")
        text = (proc.stdout + proc.stderr).strip()
        if len(text) > self._max_output:
            text = text[: self._max_output] + "\n... (salida truncada)"
        return ExecutionResult(proc.returncode == 0, text)
