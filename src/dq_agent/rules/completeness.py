import polars as pl

from dq_agent.models import RuleResult


def null_check(df: pl.DataFrame, *, column: str, max_null_rate: float = 0.0) -> RuleResult:
    total = len(df)
    if total == 0:
        return RuleResult(rule_id="null_check", passed=True, violation_rate=0.0)
    violation_rate = df[column].null_count() / total
    return RuleResult(rule_id="null_check", passed=violation_rate <= max_null_rate, violation_rate=violation_rate)
