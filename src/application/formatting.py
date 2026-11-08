from __future__ import annotations

from collections.abc import Sequence

from domain.evidence import Evidence


def format_evidence(evidence: Sequence[Evidence]) -> str:
    """Convierte evidencia en texto para los prompts, o un marcador si no hay nada."""
    return "\n".join(f"[{e.source}] {e.text}" for e in evidence) or "(sin evidencia)"
