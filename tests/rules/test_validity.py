import polars as pl

from dq_agent.rules.validity import (
    allowed_values,
    range_check,
    regex_match,
    type_conformance,
)
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
    assert result.violation_rate is None  # never evaluated, not "clean"


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


def test_regex_match_is_full_match_not_substring():
    # an unanchored pattern must match the WHOLE value, not merely appear inside it.
    # a substring search would wrongly pass '12-AB-99' (it contains digits)
    df = pl.DataFrame({"code": ["12345", "12-AB-99", "xx"]})
    result = regex_match(df, column="code", pattern=r"\d+")
    assert result.passed is False
    assert result.violation_rate == 2 / 3  # only '12345' is all-digits


# --- type_conformance ---

def test_type_conformance_passes_on_native_numeric(orders_df):
    # amount is already a numeric column — it conforms to float trivially
    result = type_conformance(orders_df, column="amount", expected_type="float")
    assert result.passed is True
    assert result.violation_rate == 0.0


def test_type_conformance_passes_on_native_date(orders_df):
    result = type_conformance(orders_df, column="created_at", expected_type="date")
    assert result.passed is True
    assert result.violation_rate == 0.0


def test_type_conformance_flags_text_that_is_not_integer():
    # a count column loaded as text with one bad value
    df = pl.DataFrame({"count": ["1", "2", "x"]})
    result = type_conformance(df, column="count", expected_type="int")
    assert result.passed is False
    assert result.violation_rate == 1 / 3


def test_type_conformance_validates_string_dates():
    df = pl.DataFrame({"day": ["2026-01-01", "2026-01-02", "not a date"]})
    result = type_conformance(df, column="day", expected_type="date")
    assert result.passed is False
    assert result.violation_rate == 1 / 3


def test_type_conformance_floats_stored_as_text():
    df = pl.DataFrame({"weight": ["1.5", "2.0", "bad"]})
    result = type_conformance(df, column="weight", expected_type="float")
    assert result.passed is False
    assert result.violation_rate == 1 / 3


def test_type_conformance_ignores_nulls():
    df = pl.DataFrame({"count": ["1", None, "2"]})
    result = type_conformance(df, column="count", expected_type="int")
    assert result.passed is True
    assert result.violation_rate == 0.0


def test_type_conformance_unsupported_type_is_error():
    df = pl.DataFrame({"x": ["1", "2"]})
    result = type_conformance(df, column="x", expected_type="complex")
    assert result.passed is False
    assert result.error is not None
    assert result.violation_rate is None  # never evaluated, not "clean"
