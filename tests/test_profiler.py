from datetime import date

import polars as pl

from dq_agent.profiler import profile, redact
from tests.conftest import NULL_CUSTOMER_IDS, TOTAL_ROWS


def _column(report, name):
    return next(c for c in report.columns if c.name == name)


# --- table stats ---

def test_profile_table_stats(orders_df):
    report = profile(orders_df, dataset="orders")
    assert report.dataset == "orders"
    assert report.table.row_count == TOTAL_ROWS
    assert report.table.duplicate_row_count == 0  # order_id 1001 repeats, but rows differ
    assert report.redacted is False


def test_profile_counts_duplicate_rows():
    df = pl.DataFrame({"a": [1, 1, 1, 2], "b": ["x", "x", "x", "y"]})
    report = profile(df, dataset="dupes")
    assert report.table.duplicate_row_count == 2  # three identical rows -> two surplus


def test_schema_fingerprint_stable_and_sensitive(orders_df):
    fingerprint = profile(orders_df, dataset="orders").table.schema_fingerprint
    assert profile(orders_df, dataset="other").table.schema_fingerprint == fingerprint
    renamed = profile(orders_df.rename({"amount": "total"}), dataset="orders")
    assert renamed.table.schema_fingerprint != fingerprint


# --- column stats ---

def test_profile_null_rate(orders_df):
    report = profile(orders_df, dataset="orders")
    assert _column(report, "customer_id").null_rate == NULL_CUSTOMER_IDS / TOTAL_ROWS
    assert _column(report, "order_id").null_rate == 0.0


def test_profile_uniqueness_ratio(orders_df):
    report = profile(orders_df, dataset="orders")
    assert _column(report, "order_id").uniqueness_ratio == 19 / 20  # one duplicate id
    assert _column(report, "customer_id").uniqueness_ratio == 1.0  # nulls excluded


def test_profile_numeric_column(orders_df):
    amount = _column(profile(orders_df, dataset="orders"), "amount")
    assert amount.min == -50.0
    assert amount.max == 300.0
    assert amount.numeric is not None
    assert amount.numeric.mean is not None


def test_profile_temporal_min_max_are_isoformat(orders_df):
    created = _column(profile(orders_df, dataset="orders"), "created_at")
    assert created.min == "2020-01-15"
    assert created.max == "2026-06-05"
    assert created.numeric is None


def test_profile_string_column_has_no_min_max(orders_df):
    status = _column(profile(orders_df, dataset="orders"), "status")
    assert status.min is None
    assert status.max is None


def test_profile_top_values(orders_df):
    status = _column(profile(orders_df, dataset="orders"), "status")
    assert len(status.top_values) == 5  # shipped, delivered, pending, cancelled, refunded
    assert status.top_values[0].count >= status.top_values[-1].count


# --- semantic hints ---

def test_semantic_hints(orders_df):
    report = profile(orders_df, dataset="orders")
    expected = {
        "order_id": "id",
        "customer_id": "id",
        "email": "email",  # 19/20 well-formed clears the threshold despite the dirt
        "phone": "phone",
        "created_at": "date",
        "status": None,
        "amount": None,
    }
    assert {c.name: c.semantic_hint for c in report.columns} == expected


def test_date_hint_on_string_typed_dates():
    df = pl.DataFrame({"event_day": ["2026-01-01", "2026-01-02", "not a date"]})
    report = profile(df, dataset="events")
    assert _column(report, "event_day").semantic_hint is None  # 2/3 below threshold

    df = pl.DataFrame({"event_day": ["2026-01-01", "2026-01-02", "2026-01-03"]})
    report = profile(df, dataset="events")
    assert _column(report, "event_day").semantic_hint == "date"


# --- redaction ---

def test_redact_strips_top_values(orders_df):
    report = profile(orders_df, dataset="orders")
    safe = redact(report)
    assert safe.redacted is True
    assert all(c.top_values is None for c in safe.columns)
    # aggregates survive redaction
    assert _column(safe, "amount").min == -50.0
    assert _column(safe, "customer_id").null_rate == NULL_CUSTOMER_IDS / TOTAL_ROWS


def test_redact_does_not_mutate_original(orders_df):
    report = profile(orders_df, dataset="orders")
    redact(report)
    assert report.redacted is False
    assert all(c.top_values is not None for c in report.columns)


def test_redacted_report_contains_no_raw_string_values(orders_df):
    payload = redact(profile(orders_df, dataset="orders")).model_dump_json()
    for raw in ("alice@example.com", "not-an-email", "C001", "+1-555-0101", "shipped"):
        assert raw not in payload


# --- edge cases ---

def test_profile_empty_dataset_is_valid_report(orders_df):
    # unlike the engine, the profiler must describe emptiness, not refuse it
    report = profile(orders_df.head(0), dataset="orders")
    assert report.table.row_count == 0
    assert report.table.duplicate_row_count == 0
    for column in report.columns:
        assert column.null_rate is None
        assert column.uniqueness_ratio is None
        assert column.top_values == []


def test_profile_all_null_column():
    df = pl.DataFrame({"x": pl.Series([None, None], dtype=pl.String)})
    column = _column(profile(df, dataset="nulls"), "x")
    assert column.null_rate == 1.0
    assert column.uniqueness_ratio is None
    assert column.semantic_hint is None


def test_profile_is_deterministic(orders_df):
    as_of = date(2026, 6, 12)
    a = profile(orders_df, dataset="orders", profiled_at=as_of)
    b = profile(orders_df, dataset="orders", profiled_at=as_of)
    assert a == b


def test_hint_sampling_hook(orders_df):
    report = profile(orders_df, dataset="orders", hint_sample_rows=5)
    # exact stats are unaffected by sampling; hints still computed
    assert report.table.row_count == TOTAL_ROWS
    assert _column(report, "email").semantic_hint == "email"


def test_report_serializes_to_json(orders_df):
    payload = profile(orders_df, dataset="orders").model_dump_json()
    assert '"row_count":20' in payload
