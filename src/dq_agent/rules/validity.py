import polars as pl

from dq_agent.models import RuleResult


def range_check(
    df: pl.DataFrame, *, column: str, min_val: float | None = None, max_val: float | None = None
) -> RuleResult:
    if min_val is None and max_val is None:
        return RuleResult(
            rule_id="range_check",
            passed=False,
            violation_rate=0.0,
            error="range_check requires at least one of min_val or max_val",
        )

    conditions = []
    if min_val is not None:
        conditions.append(pl.col(column) < min_val)
    if max_val is not None:
        conditions.append(pl.col(column) > max_val)

    violation_expr = conditions[0]
    for cond in conditions[1:]:
        violation_expr = violation_expr | cond

    violation_count = df.select(violation_expr.fill_null(False).alias("v"))["v"].sum()
    return RuleResult(
        rule_id="range_check",
        passed=violation_count == 0,
        violation_rate=violation_count / len(df),
    )


def allowed_values(df: pl.DataFrame, *, column: str, values: list) -> RuleResult:
    violation_count = df.select(
        (~pl.col(column).is_in(values) & pl.col(column).is_not_null()).alias("v")
    )["v"].sum()
    return RuleResult(
        rule_id="allowed_values",
        passed=violation_count == 0,
        violation_rate=violation_count / len(df),
    )


def regex_match(df: pl.DataFrame, *, column: str, pattern: str) -> RuleResult:
    # full-match semantics: the whole value must match, not merely contain the pattern.
    # str.contains is a regex *search*, so we anchor; the non-capturing group keeps any
    # top-level alternation (a|b) inside the anchors. Re-anchoring an already-anchored
    # pattern is harmless.
    anchored = f"^(?:{pattern})$"
    violation_count = df.select(
        (~pl.col(column).str.contains(anchored) & pl.col(column).is_not_null()).alias("v")
    )["v"].sum()
    return RuleResult(
        rule_id="regex_match",
        passed=violation_count == 0,
        violation_rate=violation_count / len(df),
    )
