from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class RuleResult(BaseModel):
    rule_id: str
    passed: bool
    violation_rate: float
    error: str | None = None


class ContractRule(BaseModel):
    rule_id: str
    params: dict[str, Any]


class Contract(BaseModel):
    dataset: str
    approved_at: datetime | None = None
    rules: list[ContractRule]
