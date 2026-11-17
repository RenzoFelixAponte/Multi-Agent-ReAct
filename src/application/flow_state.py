from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from domain.audit_verdict import AuditVerdict
from domain.evidence import Evidence
from domain.report import Report


class FlowState(TypedDict, total=False):
    """Estado compartido del grafo entre nodos."""

    question: str
    # Dos ramas paralelas (kb y web) escriben aquí; el reductor concatena sus resultados.
    prefetched: Annotated[list[Evidence], operator.add]
    report: Optional[Report]
    verdict: Optional[AuditVerdict]
    feedback: str
    rounds: int
    final: str
