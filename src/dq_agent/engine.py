"""Run-time half of the tool: executes an approved contract against a dataset.

`run()` is the only entry point that executes rules. It is deterministic and has no
LLM involvement — this is what a scheduled pipeline imports and calls on every run,
gating on the returned results. The profiler plays no part here; it belongs to
scoping time (see profiler.py).

Pre-flight checks live in the engine, not in individual rules and not in callers:
the engine is the single gatekeeper of "is this execution valid at all". Today that
means rejecting empty datasets — every rule would otherwise pass vacuously, a silent
failure mode. Phase 3 adds contract-approval (`approved_at`) and schema-drift checks
in the same place. Rules stay pure measurements; callers cannot forget the checks.
"""

from __future__ import annotations

import polars as pl

from dq_agent.models import Contract, RuleResult
from dq_agent.registry import Registry


def run(contract: Contract, df: pl.DataFrame, registry: Registry) -> list[RuleResult]:
    # an empty dataset cannot vacuously satisfy any rule — fail all of them up front
    if df.is_empty():
        return [
            RuleResult(
                rule_id=contract_rule.rule_id,
                passed=False,
                violation_rate=None,
                error="cannot evaluate: dataset is empty",
            )
            for contract_rule in contract.rules
        ]

    results: list[RuleResult] = []

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

        results.append(result)

    return results
