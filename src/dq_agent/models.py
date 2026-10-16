"""The contract: the scoping agent's only product, and what data-flow runs.

A contract is a Great Expectations 0.18.2 suite in a small envelope (see `gx_contract.py`, which holds the rules data-flow applies when it loads one). Expectation names and kwargs are GX-native. `approval` is stamped by the human approval gate; data-flow refuses a contract without it. Contracts are stored one per table as `<schema>.<table>.yaml`.

Validity is defined in one place: `problems()` delegates to the data-flow allowlist, so whatever the agent saves is exactly what data-flow will accept.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from dq_agent import gx_contract


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Notes(_Strict):
    format: Literal["string", "markdown"] = "string"
    content: str


Severity = Literal["error", "warning"]


class Meta(_Strict):
    # What a failure does to the pipeline run: "error" stops it, "warning" is recorded and the run completes
    severity: Severity = gx_contract.DEFAULT_SEVERITY
    notes: Notes | None = None  # short description surfaced on the data-flow monitoring dashboard


class Expectation(_Strict):
    expectation_type: str
    kwargs: dict[str, Any]
    meta: Meta | None = None

    @property
    def severity(self) -> Severity:
        # No meta means no severity in the YAML, which data-flow reads as an error
        return self.meta.severity if self.meta else gx_contract.DEFAULT_SEVERITY


class Dataset(_Strict):
    # `schema` shadows a BaseModel attribute, so the field is schema_ under the YAML alias
    schema_: str = Field(alias="schema")
    table: str

    @property
    def name(self) -> str:
        return f"{self.schema_}.{self.table}"


class Approval(_Strict):
    approved_at: datetime
    approved_by: str


class Suite(_Strict):
    name: str
    expectations: list[Expectation]


class Contract(_Strict):
    contract_version: int = gx_contract.CONTRACT_VERSION
    gx_version: str = gx_contract.GX_VERSION
    dataset: Dataset
    approval: Approval | None = None
    suite: Suite

    @classmethod
    def draft(cls, schema: str, table: str, expectations: list[Expectation]) -> Contract:
        dataset = Dataset(schema=schema, table=table)
        return cls(dataset=dataset, suite=Suite(name=dataset.name, expectations=expectations))

    def approve(self, approved_by: str) -> Approval:
        self.approval = Approval(approved_at=datetime.now(UTC), approved_by=approved_by)
        return self.approval

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, source: str | Path) -> Contract:
        text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
        return cls.model_validate(yaml.safe_load(text))

    def problems(self) -> list[str]:
        """Everything data-flow would reject, except a missing approval (the gate adds it)."""
        data = self.to_dict()
        data.setdefault("approval", {"approved_at": "1970-01-01T00:00:00Z", "approved_by": "pending"})
        return gx_contract.contract_errors(data)
