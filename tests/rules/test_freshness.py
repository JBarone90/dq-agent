import datetime

from dq_agent.rules.freshness import freshness
from tests.conftest import FRESHNESS_THRESHOLD_DAYS, STALE_ORDERS, TOTAL_ROWS

# Pinned reference date: clean rows span 2026-05-15..2026-06-05, stale row is 2020-01-15.
AS_OF = datetime.date(2026, 6, 11)


def test_freshness_fails_on_stale_row(orders_df):
    result = freshness(
        orders_df, column="created_at", max_days=FRESHNESS_THRESHOLD_DAYS, as_of=AS_OF
    )
    assert result.passed is False
    assert result.violation_rate == STALE_ORDERS / TOTAL_ROWS


def test_freshness_passes_with_large_threshold(orders_df):
    result = freshness(orders_df, column="created_at", max_days=9999, as_of=AS_OF)
    assert result.passed is True
    assert result.violation_rate == 0.0


def test_freshness_defaults_to_today(orders_df):
    # without as_of the reference date is today; both calls must agree
    pinned = freshness(
        orders_df, column="created_at", max_days=30, as_of=datetime.date.today()
    )
    default = freshness(orders_df, column="created_at", max_days=30)
    assert default == pinned
