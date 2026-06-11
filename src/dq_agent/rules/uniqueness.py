import polars as pl

from dq_agent.models import RuleResult


def unique_check(df: pl.DataFrame, *, column: str) -> RuleResult:
    total = len(df)
    if total == 0:
        return RuleResult(rule_id="unique_check", passed=True, violation_rate=0.0)
    non_null = df[column].drop_nulls()
    duplicate_count = len(non_null) - non_null.n_unique()
    return RuleResult(
        rule_id="unique_check",
        passed=duplicate_count == 0,
        violation_rate=duplicate_count / total,
    )
