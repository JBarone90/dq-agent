import polars as pl

from dq_agent.rules.volume import min_row_count
from tests.conftest import TOTAL_ROWS


def test_min_row_count_passes_when_above_threshold(orders_df):
    result = min_row_count(orders_df, min_rows=TOTAL_ROWS - 1)
    assert result.passed is True
    assert result.violation_rate == 0.0


def test_min_row_count_passes_at_exact_threshold(orders_df):
    result = min_row_count(orders_df, min_rows=TOTAL_ROWS)
    assert result.passed is True
    assert result.violation_rate == 0.0


def test_min_row_count_fails_below_threshold(orders_df):
    result = min_row_count(orders_df, min_rows=TOTAL_ROWS + 5)
    assert result.passed is False
    assert result.violation_rate == 5 / (TOTAL_ROWS + 5)


def test_min_row_count_single_row_far_below_threshold():
    df = pl.DataFrame({"x": [1]})
    result = min_row_count(df, min_rows=100)
    assert result.passed is False
    assert result.violation_rate == 99 / 100
