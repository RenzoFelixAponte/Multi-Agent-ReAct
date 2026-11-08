from __future__ import annotations

from typing import Optional, TypedDict

from domain.audit_verdict import AuditVerdict
from domain.report import Report


class FlowState(TypedDict, total=False):
    """Estado compartido del grafo entre nodos."""

    question: str
    report: Optional[Report]
    verdict: Optional[AuditVerdict]
    feedback: str
    rounds: int
    final: str
