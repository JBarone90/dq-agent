import datetime

import polars as pl

from dq_agent.models import RuleResult


def freshness(
    df: pl.DataFrame,
    *,
    column: str,
    max_days: int,
    as_of: datetime.date | None = None,
) -> RuleResult:
    total = len(df)
    if total == 0:
        return RuleResult(rule_id="freshness", passed=True, violation_rate=0.0)
    reference = as_of if as_of is not None else datetime.date.today()
    cutoff = reference - datetime.timedelta(days=max_days)
    violation_count = df.select(
        (pl.col(column) < cutoff).fill_null(False).alias("v")
    )["v"].sum()
    return RuleResult(
        rule_id="freshness",
        passed=violation_count == 0,
        violation_rate=violation_count / total,
    )
