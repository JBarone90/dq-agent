import os

import polars as pl
import pytest

from dq_agent.connectors import load_csv, load_parquet, load_postgres
from dq_agent.profiler import profile
from tests.conftest import DATA_DIR, TOTAL_ROWS

POSTGRES_URI = os.environ.get("DQ_TEST_POSTGRES_URI")


def test_load_csv_parses_dates():
    df = load_csv(DATA_DIR / "orders.csv")
    assert len(df) == TOTAL_ROWS
    assert df["created_at"].dtype == pl.Date


def test_load_parquet_roundtrip(orders_df, tmp_path):
    path = tmp_path / "orders.parquet"
    orders_df.write_parquet(path)
    assert load_parquet(path).equals(orders_df)


def test_load_postgres_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        load_postgres("postgresql://localhost/db")
    with pytest.raises(ValueError, match="exactly one"):
        load_postgres("postgresql://localhost/db", table="orders", query="SELECT 1")


def test_load_postgres_rejects_malformed_table_name():
    with pytest.raises(ValueError, match="invalid table name"):
        load_postgres("postgresql://localhost/db", table="orders; DROP TABLE x")


@pytest.mark.skipif(POSTGRES_URI is None, reason="DQ_TEST_POSTGRES_URI not set")
def test_postgres_and_csv_produce_identical_report_format(orders_df):
    # Phase 2 exit criterion: same structured report from a file and a Postgres table
    pg_df = load_postgres(POSTGRES_URI, table="orders")
    pg_report = profile(pg_df, dataset="orders")
    csv_report = profile(orders_df, dataset="orders")

    assert pg_report.table.row_count == csv_report.table.row_count
    assert pg_report.table.duplicate_row_count == csv_report.table.duplicate_row_count

    assert [c.name for c in pg_report.columns] == [c.name for c in csv_report.columns]
    for pg_col, csv_col in zip(pg_report.columns, csv_report.columns):
        # dtype strings may legitimately differ across sources (e.g. Int32 vs Int64);
        # every measured statistic must not
        assert pg_col.null_rate == csv_col.null_rate, pg_col.name
        assert pg_col.uniqueness_ratio == csv_col.uniqueness_ratio, pg_col.name
        assert pg_col.min == csv_col.min, pg_col.name
        assert pg_col.max == csv_col.max, pg_col.name
        assert pg_col.semantic_hint == csv_col.semantic_hint, pg_col.name
