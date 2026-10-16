"""Creative mode: proposals to allow a Great Expectations expectation the allowlist lacks.

When an owner needs a check no allowed expectation covers, the agent can look through the expectations GX 0.18.2 ships but data-flow does not allow yet (`unlisted_expectations` — the diff between GX's registry and `ALLOWED_EXPECTATIONS`) and write a proposal for a developer to review. A proposal is a hypothesis, never executed: it goes to `proposals/`, not into the draft contract. Promoting it means adding the name to `ALLOWED_EXPECTATIONS` (plus optional notes in `registry/expectations.yaml`) and running it once against a real table; the proposal file then gets `status: promoted` or is deleted.

`sql_support` is a hint, not a verdict. It is "supported" when every metric the expectation declares has a SQLAlchemy implementation in GX's metric registry, and "unverified" otherwise — some expectations (e.g. `expect_column_values_to_be_in_type_list`) choose their metrics at run time and do work on SQL despite reading as unverified. The real-table run at promotion decides.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from dq_agent import gx_contract

DEFAULT_PROPOSALS_DIR = Path("proposals")


class UnlistedExpectation(BaseModel):
    expectation_type: str
    description: str  # first line of the GX docstring
    sql_support: Literal["supported", "unverified"]


class Proposal(BaseModel):
    status: Literal["proposed", "promoted", "rejected"] = "proposed"
    proposed_at: datetime
    dataset: str  # schema.table the need came up for
    expectation_type: str
    gx_description: str
    sql_support: Literal["supported", "unverified"]
    rationale: str  # the owner's need, in plain English
    example_kwargs: dict[str, Any]  # validated against GX and the data-flow policy
    example_pass: str  # invented illustrations — never real cell values
    example_fail: str

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def unlisted_expectations() -> list[UnlistedExpectation]:
    """Every expectation GX 0.18.2 registers that the data-flow allowlist does not include."""
    import great_expectations.expectations.core  # noqa: F401 — registers the core expectations
    from great_expectations.expectations import registry as gx_registry

    return [
        UnlistedExpectation(
            expectation_type=name,
            description=_first_line(name),
            sql_support=_sql_support(name),
        )
        for name in sorted(gx_registry._registered_expectations)
        if name not in gx_contract.ALLOWED_EXPECTATIONS
    ]


def proposal_errors(expectation_type: str, example_kwargs: dict[str, Any]) -> list[str]:
    """Why a proposal cannot be written: not a GX expectation, already allowed, or kwargs
    that GX or the data-flow policy would reject."""
    if expectation_type in gx_contract.ALLOWED_EXPECTATIONS:
        return [f"'{expectation_type}' is already allowed — put it in the contract instead"]
    if expectation_type not in {u.expectation_type for u in unlisted_expectations()}:
        return [f"'{expectation_type}' is not a Great Expectations 0.18.2 expectation"]
    return gx_contract.kwarg_errors(expectation_type, example_kwargs)


def write_proposal(proposal: Proposal, proposals_dir: Path = DEFAULT_PROPOSALS_DIR) -> Path:
    proposals_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{proposal.proposed_at.date().isoformat()}_{proposal.expectation_type}_{proposal.dataset}"
    path = proposals_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', stem)}.yaml"
    suffix = 2
    while path.exists():  # never overwrite an earlier proposal
        path = path.with_name(f"{path.stem.rsplit('__', 1)[0]}__{suffix}.yaml")
        suffix += 1
    path.write_text(proposal.to_yaml(), encoding="utf-8")
    return path


def new_proposal(**fields: Any) -> Proposal:
    name = fields["expectation_type"]
    return Proposal(
        proposed_at=datetime.now(UTC),
        gx_description=_first_line(name),
        sql_support=_sql_support(name),
        **fields,
    )


def _first_line(name: str) -> str:
    from great_expectations.expectations.registry import get_expectation_impl

    return (get_expectation_impl(name).__doc__ or name).strip().splitlines()[0]


def _sql_support(name: str) -> Literal["supported", "unverified"]:
    from great_expectations.expectations import registry as gx_registry

    impl = gx_registry.get_expectation_impl(name)
    metrics = list(getattr(impl, "metric_dependencies", ()) or ())
    if getattr(impl, "map_metric", None):
        metrics.append(impl.map_metric + ".condition")
    providers = [gx_registry._registered_metrics.get(m, {}).get("providers", {}) for m in metrics]
    if metrics and all("SqlAlchemyExecutionEngine" in p for p in providers):
        return "supported"
    return "unverified"
