from dq_agent.rules.freshness import freshness
from tests.conftest import FRESHNESS_THRESHOLD_DAYS, STALE_ORDERS, TOTAL_ROWS


def test_freshness_fails_on_stale_row(orders_df):
    result = freshness(orders_df, column="created_at", max_days=FRESHNESS_THRESHOLD_DAYS)
    assert result.passed is False
    assert result.violation_rate == STALE_ORDERS / TOTAL_ROWS


def test_freshness_passes_with_large_threshold(orders_df):
    result = freshness(orders_df, column="created_at", max_days=9999)
    assert result.passed is True
    assert result.violation_rate == 0.0
