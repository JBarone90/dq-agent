"""Shared models: the contract that gates execution and the result schema pipelines consume.

A Contract is the product of scoping plus human approval — the Phase 3 scoping agent
produces it through the approval gate; contracts can also be hand-written YAML (see
contracts/examples/). Approval is recorded as `approved_at` + `approved_by`; the engine
refuses contracts without `approved_at`. `columns` snapshots the schema the contract was
scoped against (name -> dtype string) so the engine can detect drift before running.

A RuleResult is the engine's output unit: `violation_rate` is a measured 0-1 rate, or
None when the rule was never evaluated (`error` set). `severity` is the *effective*
severity stamped by the engine — the contract's per-rule override if present, else the
registry default — so downstream consumers can act on a result alone (fail on `error`,
log on `warning`) without re-joining the registry.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class RuleResult(BaseModel):
    rule_id: str
    passed: bool
    # None means the rule was never evaluated (error is set); 0.0 means measured clean
    violation_rate: float | None
    error: str | None = None
    severity: str | None = None  # stamped by the engine, not by rule functions


class ContractRule(BaseModel):
    rule_id: str
    params: dict[str, Any]
    # severity is a property of the rule *in context*: None defers to the registry default
    severity: str | None = None


class Contract(BaseModel):
    dataset: str
    approved_at: datetime | None = None
    approved_by: str | None = None
    # schema snapshot at approval time (column name -> dtype string); None skips drift check
    columns: dict[str, str] | None = None
    rules: list[ContractRule]

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json", exclude_none=True), sort_keys=False
        )

    @classmethod
    def from_yaml(cls, source: str | Path) -> Contract:
        text = source.read_text() if isinstance(source, Path) else source
        return cls.model_validate(yaml.safe_load(text))
