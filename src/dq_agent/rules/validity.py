import polars as pl

from dq_agent.models import RuleResult

# expected_type -> the Polars dtype non-string values are cast to. Date/datetime are
# parsed from text via str.to_date/to_datetime (a plain cast does not parse formats).
_NUMERIC_TARGETS = {"int": pl.Int64, "float": pl.Float64}
_TEMPORAL_TYPES = {"date", "datetime"}
_SUPPORTED_TYPES = sorted({*_NUMERIC_TARGETS, *_TEMPORAL_TYPES})


def range_check(
    df: pl.DataFrame, *, column: str, min_val: float | None = None, max_val: float | None = None
) -> RuleResult:
    if min_val is None and max_val is None:
        # never evaluated: no measurement was taken, so violation_rate is None
        # (reserved for un-evaluated results), not 0.0 which would read as "clean"
        return RuleResult(
            rule_id="range_check",
            passed=False,
            violation_rate=None,
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


def type_conformance(df: pl.DataFrame, *, column: str, expected_type: str) -> RuleResult:
    # The acting-on-it half of the profiler's inferred_dtype finding: when a column is
    # stored as text but should encode a narrower type (dates, counts, weights loaded as
    # strings), the contract declares expected_type and this rule fails each value that
    # does not parse as it. Detection only routes attention; this validates and surfaces
    # the offending rows. A column already of the expected type conforms trivially.
    if expected_type not in _SUPPORTED_TYPES:
        # never evaluated: bailed before measuring the column, so violation_rate is None
        return RuleResult(
            rule_id="type_conformance",
            passed=False,
            violation_rate=None,
            error=f"unsupported expected_type {expected_type!r}; one of {_SUPPORTED_TYPES}",
        )

    non_null = df[column].drop_nulls()
    if len(non_null) == 0:
        # nulls are absent values, not type violations (other validity rules ignore them)
        return RuleResult(rule_id="type_conformance", passed=True, violation_rate=0.0)

    failures = _coercion_failures(non_null, expected_type)
    return RuleResult(
        rule_id="type_conformance",
        passed=failures == 0,
        violation_rate=failures / len(df),
    )


def _coercion_failures(non_null: pl.Series, expected_type: str) -> int:
    """How many non-null values fail to parse as `expected_type` (parse failures become
    null under a non-strict coercion). Already-typed columns report zero."""
    if expected_type in _TEMPORAL_TYPES:
        if non_null.dtype.is_temporal():
            return 0
        if non_null.dtype != pl.String:
            return len(non_null)  # non-text, non-temporal cannot encode a date
        parse = non_null.str.to_date if expected_type == "date" else non_null.str.to_datetime
        try:
            return parse(strict=False).null_count()
        except pl.exceptions.PolarsError:
            return len(non_null)
    try:
        return non_null.cast(_NUMERIC_TARGETS[expected_type], strict=False).null_count()
    except pl.exceptions.PolarsError:
        return len(non_null)
