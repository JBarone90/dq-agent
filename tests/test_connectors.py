import polars as pl
import pytest

from dq_agent import connectors
from dq_agent.connectors import (
    DEFAULT_DSN_ENV,
    column_bounds,
    estimate_row_count,
    is_valid_table,
    load_csv,
    load_parquet,
    load_postgres,
    load_postgres_profiling,
    quote_table,
    resolve_dsn,
)
from tests.conftest import DATA_DIR, TOTAL_ROWS


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


@pytest.mark.parametrize("name,ok", [
    ("orders", True),            # bare
    ("public.orders", True),     # schema-qualified
    ("_t._c", True),             # leading underscores
    ("", False),
    ("1orders", False),          # cannot start with a digit
    ("public.orders.x", False),  # only one level of qualification
    ("orders; DROP TABLE x", False),
])
def test_is_valid_table(name, ok):
    # the single source of truth shared by the loaders and the UI's client-side check
    assert is_valid_table(name) is ok


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
    # quoted, as GX quotes them, so a mixed-case name is not folded to lower case
    assert captured["query"] == 'SELECT min("amount") AS lo, max("amount") AS hi FROM "public"."orders"'


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
    # resolved the way SELECT * FROM resolves it, so the estimate is for the table that is read
    assert """to_regclass('"orders"')""" in queries[0]


def test_estimate_row_count_unknown_returns_none(monkeypatch):
    # an un-analyzed table reports reltuples = -1 -> treated as "unknown", not zero
    _fake_pg(monkeypatch, estimate=-1)
    assert estimate_row_count("postgresql://localhost/db", "orders") is None


def test_profiling_load_full_when_under_cap(monkeypatch):
    queries = _fake_pg(monkeypatch, estimate=100)
    load = load_postgres_profiling("postgresql://localhost/db", table="orders", max_rows=1000)
    assert load.sampled is False
    assert load.estimated_rows == 100
    assert queries[-1] == 'SELECT * FROM "orders"'  # whole table, no sampling


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
    assert queries[-1] == 'SELECT * FROM "orders" LIMIT 50000'  # bounded, no TABLESAMPLE


# A live Postgres comparison of file and table profiles is in tests/test_postgres.py.


@pytest.mark.parametrize("name, quoted", [
    ("orders", '"orders"'),
    ("public.Orders", '"public"."Orders"'),
])
def test_quote_table_keeps_exact_case(name, quoted):
    assert quote_table(name) == quoted


def test_quote_table_rejects_a_quote_character():
    with pytest.raises(ValueError, match="invalid table name"):
        quote_table('orders"; DROP TABLE x; --')


def _libpq(monkeypatch, parsed):
    monkeypatch.setattr(connectors, "_parse_libpq_dsn", lambda conn_str: parsed)


def test_libpq_dsn_becomes_a_uri_with_escaped_credentials(monkeypatch):
    _libpq(monkeypatch, {"host": "db", "port": "5432", "dbname": "dw", "user": "me@corp", "password": "p/w:x"})
    assert connectors._normalise_postgres_uri("host=db ...") == "postgresql://me%40corp:p%2Fw%3Ax@db:5432/dw"


def test_libpq_dsn_leaves_out_what_libpq_would_default(monkeypatch):
    """No password (.pgpass) and no port are not errors; extra options ride as query parameters."""
    _libpq(monkeypatch, {"host": "db", "dbname": "dw", "user": "me", "sslmode": "require"})
    assert connectors._normalise_postgres_uri("host=db ...") == "postgresql://me@db/dw?sslmode=require"


def test_ready_uri_passes_through_unparsed(monkeypatch):
    _libpq(monkeypatch, None)  # would break if it were called
    assert connectors._normalise_postgres_uri("postgres://u@h/d") == "postgres://u@h/d"


# --- edges ------------------------------------------------------------------


def test_a_libpq_dsn_is_parsed_by_psycopg():
    pytest.importorskip("psycopg")
    assert connectors._parse_libpq_dsn("host=db port=5432 dbname=dw user=me") == {
        "host": "db", "port": "5432", "dbname": "dw", "user": "me"}


def test_a_libpq_dsn_falls_back_to_psycopg2(monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "psycopg.conninfo", None)
    extensions = types.ModuleType("psycopg2.extensions")
    extensions.parse_dsn = lambda conn_str: {"host": "db", "dbname": "dw"}
    monkeypatch.setitem(sys.modules, "psycopg2", types.ModuleType("psycopg2"))
    monkeypatch.setitem(sys.modules, "psycopg2.extensions", extensions)
    assert connectors._parse_libpq_dsn("host=db dbname=dw") == {"host": "db", "dbname": "dw"}


def test_a_libpq_dsn_without_either_driver_says_what_to_install(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "psycopg.conninfo", None)
    monkeypatch.setitem(sys.modules, "psycopg2.extensions", None)
    with pytest.raises(ImportError, match="uv sync --extra postgres"):
        connectors._parse_libpq_dsn("host=db")


def test_a_libpq_dsn_with_only_defaults_becomes_a_local_uri(monkeypatch):
    _libpq(monkeypatch, {"dbname": "dw", "sslmode": "require"})
    assert connectors._normalise_postgres_uri("dbname=dw sslmode=require") == "postgresql://localhost/dw?sslmode=require"


@pytest.mark.parametrize("query", [
    "SELECT * FROM orders; DELETE FROM orders",
    "WITH x AS (SELECT 1) UPDATE orders SET a = 1",
])
def test_a_mutating_query_is_refused(query):
    with pytest.raises(ValueError, match="forbidden"):
        load_postgres("postgresql://x/y", query=query)


def test_profiling_load_resolves_the_dsn_from_the_environment(monkeypatch):
    queries = _fake_pg(monkeypatch, estimate=10)
    monkeypatch.setenv("DATABASE_DSN__datasets_1", "postgresql://env/db")
    seen = []
    real = pl.read_database_uri
    monkeypatch.setattr(pl, "read_database_uri", lambda q, uri, engine: (seen.append(uri), real(q, uri, engine))[1])
    load = load_postgres_profiling(table="orders", max_rows=50)
    assert load.sampled is False
    assert set(seen) == {"postgresql://env/db"}
    assert queries


def test_column_bounds_of_an_empty_result(monkeypatch):
    monkeypatch.setattr(pl, "read_database_uri", lambda q, uri, engine: pl.DataFrame({"lo": [], "hi": []}))
    assert connectors.column_bounds("postgresql://x/y", "orders", "amount") == (None, None)


def test_column_bounds_refuses_a_column_that_is_not_an_identifier():
    with pytest.raises(ValueError, match="invalid column name"):
        connectors.column_bounds("postgresql://x/y", "orders", 'amount"; --')


def test_a_missing_connectorx_says_what_to_install(monkeypatch):
    def no_connectorx(query, uri, engine):
        raise ImportError("connectorx")

    monkeypatch.setattr(pl, "read_database_uri", no_connectorx)
    with pytest.raises(ImportError, match="uv sync --extra postgres"):
        load_postgres("postgresql://x/y", table="orders")
