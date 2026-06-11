import polars as pl

from dq_agent.rules.validity import allowed_values, range_check, regex_match
from tests.conftest import (
    EMAIL_PATTERN,
    INVALID_STATUSES,
    MALFORMED_EMAILS,
    NEGATIVE_AMOUNTS,
    TOTAL_ROWS,
    VALID_STATUSES,
)


# --- range_check ---

def test_range_check_fails_on_negative_amount(orders_df):
    result = range_check(orders_df, column="amount", min_val=0.0)
    assert result.passed is False
    assert result.violation_rate == NEGATIVE_AMOUNTS / TOTAL_ROWS


def test_range_check_no_bounds_returns_error():
    import polars as pl
    df = pl.DataFrame({"x": [1.0, 2.0]})
    result = range_check(df, column="x")
    assert result.passed is False
    assert result.error is not None


def test_range_check_ignores_nulls():
    df = pl.DataFrame({"x": [1.0, None, 3.0]})
    result = range_check(df, column="x", min_val=0.0, max_val=5.0)
    assert result.passed is True


def test_range_check_upper_bound():
    df = pl.DataFrame({"x": [1.0, 2.0, 100.0]})
    result = range_check(df, column="x", max_val=10.0)
    assert result.passed is False
    assert result.violation_rate == 1 / 3


# --- allowed_values ---

def test_allowed_values_fails_on_invalid_status(orders_df):
    result = allowed_values(orders_df, column="status", values=list(VALID_STATUSES))
    assert result.passed is False
    assert result.violation_rate == INVALID_STATUSES / TOTAL_ROWS


def test_allowed_values_passes_when_extended(orders_df):
    result = allowed_values(orders_df, column="status", values=list(VALID_STATUSES) + ["refunded"])
    assert result.passed is True
    assert result.violation_rate == 0.0


def test_allowed_values_ignores_nulls():
    df = pl.DataFrame({"x": ["a", "b", None]})
    result = allowed_values(df, column="x", values=["a", "b"])
    assert result.passed is True


# --- regex_match ---

def test_regex_match_fails_on_malformed_email(orders_df):
    result = regex_match(orders_df, column="email", pattern=EMAIL_PATTERN)
    assert result.passed is False
    assert result.violation_rate == MALFORMED_EMAILS / TOTAL_ROWS


def test_regex_match_passes_on_phone(orders_df):
    result = regex_match(orders_df, column="phone", pattern=r"^\+\d-\d{3}-\d{4}$")
    assert result.passed is True
    assert result.violation_rate == 0.0


def test_regex_match_ignores_nulls():
    df = pl.DataFrame({"x": ["abc", None]})
    result = regex_match(df, column="x", pattern=r"^\d+$")
    assert result.passed is False
    assert result.violation_rate == 0.5
