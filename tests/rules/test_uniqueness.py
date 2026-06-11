import polars as pl

from dq_agent.rules.uniqueness import unique_check
from tests.conftest import DUPLICATE_ORDER_IDS, TOTAL_ROWS


def test_unique_check_fails_on_duplicate_column(orders_df):
    result = unique_check(orders_df, column="order_id")
    assert result.passed is False
    assert result.violation_rate == DUPLICATE_ORDER_IDS / TOTAL_ROWS


def test_unique_check_passes_on_clean_column(orders_df):
    result = unique_check(orders_df, column="email")
    assert result.passed is True
    assert result.violation_rate == 0.0


def test_unique_check_ignores_nulls():
    # two nulls in an otherwise-unique column should not count as duplicates
    df = pl.DataFrame({"x": [1, 2, None, None]})
    result = unique_check(df, column="x")
    assert result.passed is True
    assert result.violation_rate == 0.0
