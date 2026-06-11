from __future__ import annotations

import polars as pl

from dq_agent.models import Contract, RuleResult
from dq_agent.registry import Registry


def run(contract: Contract, df: pl.DataFrame, registry: Registry) -> list[RuleResult]:
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
                    violation_rate=0.0,
                    error="; ".join(errors),
                ))
                continue

            fn = registry.resolve(rule_id)
            result = fn(df, **params)
        except Exception as exc:
            result = RuleResult(
                rule_id=rule_id,
                passed=False,
                violation_rate=0.0,
                error=str(exc),
            )

        results.append(result)

    return results
