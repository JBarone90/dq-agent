import datetime

import polars as pl

from dq_agent.models import RuleResult


def freshness(
    df: pl.DataFrame,
    *,
    column: str,
    max_days: int,
    as_of: datetime.date | str | None = None,
) -> RuleResult:
    if isinstance(as_of, str):
        as_of = datetime.date.fromisoformat(as_of)
    reference = as_of if as_of is not None else datetime.date.today()
    cutoff = reference - datetime.timedelta(days=max_days)
    violation_count = df.select(
        (pl.col(column) < cutoff).fill_null(False).alias("v")
    )["v"].sum()
    return RuleResult(
        rule_id="freshness",
        passed=violation_count == 0,
        violation_rate=violation_count / len(df),
    )
