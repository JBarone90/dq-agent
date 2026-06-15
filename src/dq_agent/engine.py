"""Run-time half of the tool: executes an approved contract against a dataset.

`run()` is the only entry point that executes rules. It is deterministic and has no
LLM involvement — this is what a scheduled pipeline imports and calls on every run,
gating on the returned results. The profiler plays no part here; it belongs to
scoping time (see profiler.py).

Pre-flight checks live in the engine, not in individual rules and not in callers:
the engine is the single gatekeeper of "is this execution valid at all". Approval
and schema drift raise — they are contract lifecycle events that route the owner
back to scoping, not data measurements. An empty dataset does not raise: the
contract is still valid, today's data is not. A column rule cannot be evaluated on
zero rows (it would divide by zero / pass vacuously) and reports an un-evaluated
result; a table-level rule like min_row_count still runs — an empty table is
exactly its concern. Rules stay pure measurements; callers cannot forget the checks.
"""

from __future__ import annotations

import polars as pl

from dq_agent.models import Contract, ContractRule, RuleResult
from dq_agent.registry import Registry


class ContractNotApprovedError(ValueError):
    """The contract has no `approved_at` — it never passed the human approval gate."""


class SchemaDriftError(ValueError):
    """The live schema no longer matches the one the contract was scoped against."""


def run(contract: Contract, df: pl.DataFrame, registry: Registry) -> list[RuleResult]:
    if contract.approved_at is None:
        raise ContractNotApprovedError(
            f"contract for dataset '{contract.dataset}' is not approved: "
            "no rule runs without an approved contract"
        )
    _check_schema_drift(contract, df)

    results: list[RuleResult] = []
    empty = df.is_empty()

    for contract_rule in contract.rules:
        rule_id = contract_rule.rule_id
        params = contract_rule.params

        try:
            errors = registry.validate_params(rule_id, params)
            if errors:
                results.append(RuleResult(
                    rule_id=rule_id,
                    passed=False,
                    violation_rate=None,
                    error="; ".join(errors),
                    severity=_effective_severity(contract_rule, registry),
                ))
                continue

            # on zero rows a column rule would divide by zero / pass vacuously, so it
            # cannot be evaluated; table-level rules still run (an empty table is their
            # concern) and report a real failure
            if empty and not _is_table_level(rule_id, registry):
                results.append(RuleResult(
                    rule_id=rule_id,
                    passed=False,
                    violation_rate=None,
                    error="cannot evaluate: dataset is empty",
                    severity=_effective_severity(contract_rule, registry),
                ))
                continue

            fn = registry.resolve(rule_id)
            result = fn(df, **params)
        except Exception as exc:
            result = RuleResult(
                rule_id=rule_id,
                passed=False,
                violation_rate=None,
                error=str(exc),
            )

        result.severity = _effective_severity(contract_rule, registry)
        results.append(result)

    return results


def _check_schema_drift(contract: Contract, df: pl.DataFrame) -> None:
    if contract.columns is None:
        return
    live = {name: str(dtype) for name, dtype in df.schema.items()}

    drifts = []
    for name in sorted(set(contract.columns) - set(live)):
        drifts.append(f"column '{name}' removed")
    for name in sorted(set(live) - set(contract.columns)):
        drifts.append(f"column '{name}' added")
    for name in sorted(set(live) & set(contract.columns)):
        # compare coarse families, not exact dtype strings, so a cross-source load
        # (CSV Int64 vs Postgres Int32) is not mistaken for drift
        if _dtype_family(live[name]) != _dtype_family(contract.columns[name]):
            drifts.append(
                f"column '{name}' changed type: {contract.columns[name]} -> {live[name]}"
            )
    if not drifts:
        return
    raise SchemaDriftError(
        f"schema drift for dataset '{contract.dataset}', contract needs re-scoping: "
        + "; ".join(drifts)
    )


def _dtype_family(dtype: str) -> str:
    """Collapse a concrete Polars dtype string to a coarse family for drift comparison."""
    s = dtype.lower()
    if s.startswith(("int", "uint")):
        return "integer"
    if s.startswith(("float", "decimal")):
        return "float"
    if s.startswith(("date", "time", "duration")):
        return "temporal"
    if s.startswith("bool"):
        return "boolean"
    if s.startswith(("str", "utf8", "categorical", "enum")):
        return "string"
    return s


def _is_table_level(rule_id: str, registry: Registry) -> bool:
    # a rule is table-level when it takes no column parameter — it measures the table,
    # not a column; the registry definition is the source of truth
    return "column" not in registry.get(rule_id).parameters


def _effective_severity(contract_rule: ContractRule, registry: Registry) -> str | None:
    if contract_rule.severity is not None:
        return contract_rule.severity
    try:
        return registry.get(contract_rule.rule_id).severity
    except KeyError:
        return None
