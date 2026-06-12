import polars as pl

from dq_agent.models import RuleResult


def min_row_count(df: pl.DataFrame, *, min_rows: int) -> RuleResult:
    shortfall = max(min_rows - len(df), 0)
    return RuleResult(
        rule_id="min_row_count",
        # violation_rate is the shortfall as a fraction of the threshold, not a per-row rate
        violation_rate=shortfall / min_rows if min_rows > 0 else 0.0,
        passed=shortfall == 0,
    )
