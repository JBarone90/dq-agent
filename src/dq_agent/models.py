"""Shared models: the contract that gates execution and the result schema pipelines consume.

A Contract is the product of scoping plus human approval — Phase 3 produces it
through the approval gate; until then contracts are hand-written YAML (see
contracts/examples/). A RuleResult is the engine's output unit: `violation_rate`
is a measured 0-1 rate, or None when the rule was never evaluated (`error` set).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class RuleResult(BaseModel):
    rule_id: str
    passed: bool
    # None means the rule was never evaluated (error is set); 0.0 means measured clean
    violation_rate: float | None
    error: str | None = None


class ContractRule(BaseModel):
    rule_id: str
    params: dict[str, Any]


class Contract(BaseModel):
    dataset: str
    approved_at: datetime | None = None
    rules: list[ContractRule]
