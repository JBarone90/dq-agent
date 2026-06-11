from pathlib import Path

import polars as pl
import pytest

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

# Known issues baked into orders.csv — tests assert against these exact counts.
#
# order_id   : 1 duplicate  — value 1001 appears in rows 1 and 2
# customer_id: 2 nulls      — rows 5, 8
# amount     : 1 null       — row 13
#              1 negative   — row 11 (-50.00)
# email      : 1 malformed  — row 15 ("not-an-email")
# status     : 1 invalid    — row 17 ("refunded" not in VALID_STATUSES)
# created_at : 1 stale      — row 19 (2020-01-15, >365 days old)

DUPLICATE_ORDER_IDS = 1
NULL_CUSTOMER_IDS = 2
NULL_AMOUNTS = 1
NEGATIVE_AMOUNTS = 1
MALFORMED_EMAILS = 1
INVALID_STATUSES = 1
STALE_ORDERS = 1

VALID_STATUSES = {"shipped", "delivered", "pending", "cancelled"}
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
FRESHNESS_THRESHOLD_DAYS = 365


@pytest.fixture(scope="session")
def synthetic_data_path() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def orders_df() -> pl.DataFrame:
    return pl.read_csv(DATA_DIR / "orders.csv", try_parse_dates=True)
