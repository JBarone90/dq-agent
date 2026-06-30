import os

import polars as pl
import pytest

from dq_agent.connectors import (
    DEFAULT_DSN_ENV,
    column_bounds,
    estimate_row_count,
    load_csv,
    load_parquet,
    load_postgres,
    load_postgres_profiling,
    resolve_dsn,
)
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


def test_load_postgres_pushes_sample_limit_down(monkeypatch):
    captured = {}

    def fake_read(query, uri, engine):
        captured["query"] = query
        return pl.DataFrame({"a": [1]})

    monkeypatch.setattr(pl, "read_database_uri", fake_read)
    load_postgres("postgresql://localhost/db", table="orders", sample_rows=100)
    assert "LIMIT 100" in captured["query"]
    assert "random()" in captured["query"]


def test_load_postgres_sample_rejects_custom_query():
    with pytest.raises(ValueError, match="sample_rows applies to 'table'"):
        load_postgres("postgresql://localhost/db", query="SELECT 1", sample_rows=100)


def test_load_postgres_rejects_non_select_query():
    with pytest.raises(ValueError, match="read-only SELECT"):
        load_postgres("postgresql://localhost/db", query="UPDATE orders SET x = 1")


def test_load_postgres_rejects_mutating_keyword_in_select():
    # a SELECT prefix is not enough; a smuggled mutation must still be rejected
    with pytest.raises(ValueError, match="forbidden"):
        load_postgres(
            "postgresql://localhost/db",
            query="SELECT 1; DROP TABLE orders",
        )


def test_column_bounds_returns_exact_min_max(monkeypatch):
    captured = {}

    def fake_read(query, uri, engine):
        captured["query"] = query
        return pl.DataFrame({"lo": [0], "hi": [4999]})

    monkeypatch.setattr(pl, "read_database_uri", fake_read)
    lo, hi = column_bounds("postgresql://localhost/db", "public.orders", "amount")
    assert (lo, hi) == (0, 4999)
    assert "min(amount)" in captured["query"] and "max(amount)" in captured["query"]


def test_column_bounds_rejects_malformed_column():
    with pytest.raises(ValueError, match="invalid column name"):
        column_bounds("postgresql://localhost/db", "orders", "amount); DROP TABLE x")


# --- DSN resolution from the environment (mirrors the model's DQ_AGENT_MODEL read) ---

def test_resolve_dsn_passes_through_ready_uri(monkeypatch):
    monkeypatch.setenv(DEFAULT_DSN_ENV, "postgresql://u:p@host:5432/db")
    assert resolve_dsn() == "postgresql://u:p@host:5432/db"


def test_resolve_dsn_raises_when_env_unset(monkeypatch):
    monkeypatch.delenv(DEFAULT_DSN_ENV, raising=False)
    with pytest.raises(KeyError, match=DEFAULT_DSN_ENV):
        resolve_dsn()


def test_loader_resolves_uri_from_env_when_omitted(monkeypatch):
    monkeypatch.setenv(DEFAULT_DSN_ENV, "postgresql://u:p@host:5432/db")
    captured = {}

    def fake_read(query, uri, engine):
        captured["uri"] = uri
        return pl.DataFrame({"a": [1]})

    monkeypatch.setattr(pl, "read_database_uri", fake_read)
    load_postgres(table="orders")  # no uri -> resolved from env
    assert captured["uri"] == "postgresql://u:p@host:5432/db"


# --- adaptive sizing: planner estimate + TABLESAMPLE ---

def _fake_pg(monkeypatch, *, estimate):
    """Stub pl.read_database_uri: answer planner-estimate queries with `estimate`
    (None -> no row, simulating an un-analyzed table) and any other query with data.
    Returns the captured list of queries in call order."""
    queries = []

    def fake_read(query, uri, engine):
        queries.append(query)
        if "pg_class" in query:
            data = [] if estimate is None else [estimate]
            return pl.DataFrame({"estimate": data}, schema={"estimate": pl.Int64})
        return pl.DataFrame({"a": [1, 2, 3]})

    monkeypatch.setattr(pl, "read_database_uri", fake_read)
    return queries


def test_estimate_row_count_reads_planner_statistics(monkeypatch):
    queries = _fake_pg(monkeypatch, estimate=5000)
    assert estimate_row_count("postgresql://localhost/db", "orders") == 5000
    assert "pg_class" in queries[0] and "reltuples" in queries[0]


def test_estimate_row_count_unknown_returns_none(monkeypatch):
    # an un-analyzed table reports reltuples = -1 -> treated as "unknown", not zero
    _fake_pg(monkeypatch, estimate=-1)
    assert estimate_row_count("postgresql://localhost/db", "orders") is None


def test_profiling_load_full_when_under_cap(monkeypatch):
    queries = _fake_pg(monkeypatch, estimate=100)
    load = load_postgres_profiling("postgresql://localhost/db", table="orders", max_rows=1000)
    assert load.sampled is False
    assert load.estimated_rows == 100
    assert queries[-1] == "SELECT * FROM orders"  # whole table, no sampling


def test_profiling_load_tablesamples_when_over_cap(monkeypatch):
    queries = _fake_pg(monkeypatch, estimate=10_000_000)
    load = load_postgres_profiling("postgresql://localhost/db", table="orders", max_rows=50_000)
    assert load.sampled is True
    assert "TABLESAMPLE SYSTEM" in queries[-1]
    assert "REPEATABLE (0)" in queries[-1]
    assert "LIMIT 50000" in queries[-1]


def test_profiling_load_falls_back_to_limit_when_estimate_unknown(monkeypatch):
    queries = _fake_pg(monkeypatch, estimate=None)
    load = load_postgres_profiling("postgresql://localhost/db", table="orders", max_rows=50_000)
    assert load.sampled is True
    assert load.estimated_rows is None
    assert queries[-1] == "SELECT * FROM orders LIMIT 50000"  # bounded, no TABLESAMPLE


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
