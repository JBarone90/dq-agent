"""Contracts and profiling against a real Postgres: run where one is reachable (e.g. at work).

Set DQ_TEST_POSTGRES_URI to a postgresql:// URI for a database you may create a schema in. The
module loads data/synthetic/orders.csv into a fresh `dq_agent_test_<random>` schema, runs the
tests, and drops the schema again. Everything skips when the URI is unset.

These are the checks a local GX run cannot make: Postgres evaluates the regex (POSIX classes,
not Python's), `type_list` is matched against the column's declared SQL type, and the
`$PARAMETER` date bounds are compared with a DATE column inside the database.
"""

import datetime
import os
import uuid

import pytest

URI = os.environ.get("DQ_TEST_POSTGRES_URI")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(URI is None, reason="DQ_TEST_POSTGRES_URI not set")]

pytest.importorskip("great_expectations")
sqlalchemy = pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg2")  # GX 0.18.2's Postgres driver
pytest.importorskip("connectorx")  # the agent's profiling connector
pd = pytest.importorskip("pandas")

from dq_agent import connectors, gx_contract
from dq_agent.profiler import profile
from tests.conftest import DATA_DIR, TOTAL_ROWS

CSV = DATA_DIR / "orders.csv"
DDL = """CREATE TABLE {table} (
    order_id INTEGER, customer_id TEXT, email TEXT, amount DOUBLE PRECISION,
    status TEXT, created_at DATE, phone TEXT)"""


def _gx_uri(uri: str) -> str:
    """GX 0.18.2 on SQLAlchemy 1.4 needs the driver named in the URI."""
    return "postgresql+psycopg2://" + uri.split("://", 1)[1]


@pytest.fixture(scope="module")
def schema():
    name = f"dq_agent_test_{uuid.uuid4().hex[:8]}"
    engine = sqlalchemy.create_engine(_gx_uri(URI))
    frame = pd.read_csv(CSV, parse_dates=["created_at"], dtype={"customer_id": object})
    frame["created_at"] = frame["created_at"].dt.date
    rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
    columns = ", ".join(frame.columns)
    values = ", ".join(f":{c}" for c in frame.columns)
    # SQLAlchemy 1.4 (what GX 0.18.2 needs) is too old for pandas' to_sql, so insert directly
    with engine.begin() as connection:
        connection.execute(sqlalchemy.text(f'CREATE SCHEMA "{name}"'))
        connection.execute(sqlalchemy.text(DDL.format(table=f'"{name}"."orders"')))
        connection.execute(sqlalchemy.text(f'INSERT INTO "{name}"."orders" ({columns}) VALUES ({values})'), rows)
    yield name
    with engine.begin() as connection:
        connection.execute(sqlalchemy.text(f'DROP SCHEMA "{name}" CASCADE'))
    engine.dispose()


def _contract(schema, *expectations):
    return {
        "contract_version": 1, "gx_version": "0.18.2",
        "dataset": {"schema": schema, "table": "orders"},
        "approval": {"approved_at": "2026-10-16T09:00:00Z", "approved_by": "pytest"},
        "suite": {"name": f"{schema}.orders", "expectations": [
            {"expectation_type": kind, "kwargs": kwargs, "meta": {"severity": severity}}
            for kind, kwargs, severity in expectations]},
    }


def _run(contract):
    import great_expectations as gx

    result = gx_contract.run_contract(gx.get_context(mode="ephemeral"), contract, connection_string=_gx_uri(URI),
                                      checkpoint_name="pytest", run_name="pytest", raise_on_failure=False)
    return {r.expectation_config.expectation_type + ":" + str(gx_contract._columns_label(r.expectation_config.kwargs)): r
            for v in result.list_validation_results() for r in v.results}, result


def _days_ago(day: datetime.date) -> int:
    return (datetime.date.today() - day).days


NEWEST = datetime.date(2026, 6, 5)  # the newest created_at in orders.csv
OLDEST = datetime.date(2020, 1, 15)  # the stale row


def _days(n):
    return {"$PARAMETER": f"now() - timedelta(days={n})"}


def test_the_example_checks_find_each_known_issue(schema):
    results, run = _run(_contract(
        schema,
        ("expect_column_values_to_not_be_null", {"column": "customer_id"}, "error"),
        ("expect_column_values_to_be_unique", {"column": "order_id"}, "error"),
        ("expect_column_values_to_be_between", {"column": "amount", "min_value": 0}, "error"),
        ("expect_column_values_to_be_in_set",
         {"column": "status", "value_set": ["shipped", "delivered", "pending", "cancelled"]}, "error"),
        # the example contract's regex, POSIX classes as Postgres reads them
        ("expect_column_values_to_match_regex",
         {"column": "email", "regex": r"^[^@[:space:]]+@[^@[:space:]]+\.[^@[:space:]]+$"}, "warning"),
        ("expect_table_row_count_to_be_between", {"min_value": TOTAL_ROWS, "max_value": TOTAL_ROWS}, "warning"),
    ))
    unexpected = {key: r.result.get("unexpected_count") for key, r in results.items()}
    assert not any(gx_contract.could_not_run(r) for r in results.values()), gx_contract.summarize(run)
    assert unexpected == {
        "expect_column_values_to_not_be_null:customer_id": 2,
        "expect_column_values_to_be_unique:order_id": 2,
        "expect_column_values_to_be_between:amount": 1,
        "expect_column_values_to_be_in_set:status": 1,
        "expect_column_values_to_match_regex:email": 1,
        "expect_table_row_count_to_be_between:None": None,
    }
    assert results["expect_table_row_count_to_be_between:None"].success


def test_freshness_and_stale_rows_evaluate_in_postgres(schema):
    """The $PARAMETER bound is evaluated by GX and compared with a DATE column in the database."""
    fresh_enough = _days_ago(NEWEST) + 30
    results, run = _run(_contract(
        schema,
        ("expect_column_max_to_be_between", {"column": "created_at", "min_value": _days(fresh_enough)}, "warning"),
        ("expect_column_max_to_be_between", {"column": "created_at", "min_value": _days(max(_days_ago(NEWEST) - 1, 1))},
         "warning"),
        ("expect_column_min_to_be_between", {"column": "created_at", "min_value": _days(_days_ago(OLDEST) - 30)},
         "warning"),
    ))
    assert not any(gx_contract.could_not_run(r) for r in results.values()), gx_contract.summarize(run)
    outcomes = [r.success for r in run.list_validation_results()[0].results]
    assert sorted(outcomes) == [False, False, True], gx_contract.summarize(run)


def test_the_newer_expectations_run_in_postgres(schema):
    results, run = _run(_contract(
        schema,
        ("expect_column_values_to_not_be_in_set", {"column": "status", "value_set": ["refunded"]}, "warning"),
        ("expect_column_values_to_not_match_regex", {"column": "email", "regex": "^not-"}, "warning"),
        ("expect_column_value_lengths_to_be_between", {"column": "customer_id", "min_value": 4, "max_value": 4},
         "warning"),
        ("expect_column_pair_values_a_to_be_greater_than_b", {"column_A": "order_id", "column_B": "amount"}, "warning"),
        ("expect_compound_columns_to_be_unique", {"column_list": ["order_id", "customer_id"]}, "warning"),
        ("expect_select_column_values_to_be_unique_within_record", {"column_list": ["email", "phone"]}, "warning"),
        ("expect_table_columns_to_match_set", {"column_set": [
            "order_id", "customer_id", "email", "amount", "status", "created_at", "phone"]}, "warning"),
        ("expect_column_max_to_be_between", {"column": "amount", "max_value": 100000}, "warning"),
    ))
    broken = [key for key, r in results.items() if gx_contract.could_not_run(r)]
    assert broken == [], gx_contract.summarize(run)
    assert not results["expect_column_values_to_not_be_in_set:status"].success  # the 'refunded' row
    assert not results["expect_column_values_to_not_match_regex:email"].success  # 'not-an-email'
    assert results["expect_table_columns_to_match_set:None"].success


@pytest.mark.parametrize("column, type_name", [
    ("order_id", "INTEGER"), ("amount", "DOUBLE_PRECISION"), ("created_at", "DATE"), ("email", "TEXT"),
])
def test_type_list_names_match_postgres_types(schema, column, type_name):
    """Which spelling GX 0.18.2 accepts for each Postgres type. If one fails, the summary shows the
    type GX observed: update the registry notes for expect_column_values_to_be_in_type_list."""
    results, run = _run(_contract(
        schema, ("expect_column_values_to_be_in_type_list", {"column": column, "type_list": [type_name]}, "error")))
    (result,) = results.values()
    assert result.success, (gx_contract.summarize(run), result.result.get("observed_value"))


def test_a_check_on_a_missing_column_could_not_run(schema):
    results, _ = _run(_contract(schema, ("expect_column_values_to_be_unique", {"column": "nope"}, "warning")))
    (result,) = results.values()
    assert gx_contract.could_not_run(result) and gx_contract.effective_severity(result) == "error"


def test_an_error_failure_raises(schema):
    import great_expectations as gx

    contract = _contract(schema, ("expect_column_values_to_be_unique", {"column": "order_id"}, "error"))
    with pytest.raises(gx_contract.DataQualityCheckFailed):
        gx_contract.run_contract(gx.get_context(mode="ephemeral"), contract, connection_string=_gx_uri(URI),
                                 checkpoint_name="pytest", run_name="pytest")


# --- the agent's side: profiling the table --------------------------------


def test_postgres_and_csv_produce_the_same_profile(schema, orders_df):
    pg_df = connectors.load_postgres(URI, table=f"{schema}.orders")
    pg_report, csv_report = profile(pg_df, dataset="orders"), profile(orders_df, dataset="orders")

    assert pg_report.table.row_count == csv_report.table.row_count == TOTAL_ROWS
    assert pg_report.table.duplicate_row_count == csv_report.table.duplicate_row_count
    assert [c.name for c in pg_report.columns] == [c.name for c in csv_report.columns]
    for pg_col, csv_col in zip(pg_report.columns, csv_report.columns, strict=True):
        # dtype strings may differ across sources (Int32 vs Int64); measured statistics must not
        assert pg_col.null_rate == csv_col.null_rate, pg_col.name
        assert pg_col.uniqueness_ratio == csv_col.uniqueness_ratio, pg_col.name
        assert pg_col.min == csv_col.min, pg_col.name
        assert pg_col.max == csv_col.max, pg_col.name
        assert pg_col.semantic_hint == csv_col.semantic_hint, pg_col.name


def test_profiling_load_of_a_small_table_is_whole(schema):
    with connectors_analyzed(schema):
        load = connectors.load_postgres_profiling(URI, table=f"{schema}.orders", max_rows=1000)
    assert load.sampled is False and load.df.height == TOTAL_ROWS


def test_profiling_load_of_a_large_table_is_sampled(schema):
    load = connectors.load_postgres_profiling(URI, table=f"{schema}.orders", max_rows=5)
    assert load.sampled is True and load.df.height <= 5


def test_column_bounds_are_exact(schema, orders_df):
    amounts = orders_df["amount"].drop_nulls()
    assert connectors.column_bounds(URI, f"{schema}.orders", "amount") == (amounts.min(), amounts.max())
    assert connectors.column_bounds(URI, f"{schema}.orders", "created_at") == (OLDEST, NEWEST)


def test_the_agent_profiles_the_table_through_its_tool(schema, registry, monkeypatch):
    pytest.importorskip("langgraph")
    from dq_agent.agents.scoping import _make_tools

    monkeypatch.setenv(connectors.DEFAULT_DSN_ENV, URI)
    tool = {t.name: t for t in _make_tools(registry)}["profile_table"]
    command = tool.invoke({"name": "profile_table", "args": {"table": f"{schema}.orders"},
                           "id": "c1", "type": "tool_call"})
    report = command.update["profile"]
    assert report["redacted"] is True and report["table"]["row_count"] == TOTAL_ROWS
    assert "not-an-email" not in command.update["messages"][0].content


class connectors_analyzed:
    """ANALYZE the table first, so the planner estimate the loader reads is current."""

    def __init__(self, schema):
        self.schema = schema

    def __enter__(self):
        engine = sqlalchemy.create_engine(_gx_uri(URI))
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text(f'ANALYZE "{self.schema}"."orders"'))
        engine.dispose()

    def __exit__(self, *exc):
        return False


