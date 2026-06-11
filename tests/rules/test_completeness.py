import polars as pl

from dq_agent.rules.completeness import null_check
from tests.conftest import NULL_AMOUNTS, NULL_CUSTOMER_IDS, TOTAL_ROWS


def test_null_check_passes_on_clean_column(orders_df):
    result = null_check(orders_df, column="order_id", max_null_rate=0.0)
    assert result.passed is True
    assert result.violation_rate == 0.0


def test_null_check_fails_on_dirty_column(orders_df):
    result = null_check(orders_df, column="customer_id", max_null_rate=0.0)
    assert result.passed is False
    assert result.violation_rate == NULL_CUSTOMER_IDS / TOTAL_ROWS


def test_null_check_passes_within_tolerance(orders_df):
    result = null_check(orders_df, column="amount", max_null_rate=NULL_AMOUNTS / TOTAL_ROWS)
    assert result.passed is True


def test_null_check_empty_dataframe():
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Int64)})
    result = null_check(df, column="x")
    assert result.passed is True
    assert result.violation_rate == 0.0
