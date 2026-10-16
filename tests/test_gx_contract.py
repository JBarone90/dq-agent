"""`dq_agent.gx_contract`: what a contract may contain, loading one, and running it with GX.

These rules are the ones data-flow applies when it loads a contract, so every verdict below is
also what data-flow would say.
"""

import copy
import datetime
import logging
import sqlite3
from pathlib import Path

import pytest
import yaml

pytest.importorskip("great_expectations")
pd = pytest.importorskip("pandas")

from dq_agent import gx_contract

VALID = {
    "contract_version": 1,
    "gx_version": "0.18.2",
    "dataset": {"schema": "synthetic", "table": "orders"},
    "approval": {"approved_at": "2026-10-16T09:00:00Z", "approved_by": "jacopo"},
    "suite": {
        "name": "synthetic.orders",
        "expectations": [
            {"expectation_type": "expect_column_values_to_be_unique", "kwargs": {"column": "order_id"},
             "meta": {"severity": "error", "notes": {"format": "string", "content": "one row per order"}}},
            {"expectation_type": "expect_column_values_to_be_between",
             "kwargs": {"column": "amount", "min_value": 0}, "meta": {"severity": "warning"}},
        ],
    },
}


def _with(expectation_type: str, kwargs: dict, meta: dict | None = None) -> dict:
    contract = copy.deepcopy(VALID)
    entry = {"expectation_type": expectation_type, "kwargs": kwargs}
    if meta is not None:
        entry["meta"] = meta
    contract["suite"]["expectations"].append(entry)
    return contract


def _changed(path: str, value) -> dict:
    contract = copy.deepcopy(VALID)
    *parents, key = path.split(".")
    node = contract
    for name in parents:
        node = node[name]
    if value is _DELETE:
        del node[key]
    else:
        node[key] = value
    return contract


def _all_of(*expectations: dict) -> dict:
    contract = copy.deepcopy(VALID)
    contract["suite"]["expectations"] = [{"expectation_type": kind, "kwargs": kwargs} for kind, kwargs in expectations]
    return contract


_DELETE = object()
_30_DAYS = {"$PARAMETER": "now() - timedelta(days=30)"}

CONTRACTS = {
    "valid": VALID,
    "not a mapping": ["a", "list"],
    "wrong contract_version": _changed("contract_version", 2),
    "wrong gx_version": _changed("gx_version", "1.0"),
    "unknown top-level key": _changed("checkpoint", {}),
    "dataset not a mapping": _changed("dataset", "orders"),
    "dataset unknown key": _changed("dataset.database", "dw"),
    "dataset quoted identifier": _changed("dataset.table", 'orders"; --'),
    "no approval": _changed("approval", _DELETE),
    "approval date only": _changed("approval.approved_at", "2026-10-16"),
    "approval naive timestamp": _changed("approval.approved_at", "2026-10-16T09:00:00"),
    "approval with offset": _changed("approval.approved_at", "2026-10-16T09:00:00+01:00"),
    "approval not a timestamp": _changed("approval.approved_at", "yesterday"),
    "approval as a YAML timestamp": _changed(
        "approval.approved_at", datetime.datetime(2026, 10, 16, 9, tzinfo=datetime.UTC)),
    "approval as a naive YAML timestamp": _changed("approval.approved_at", datetime.datetime(2026, 10, 16, 9)),
    "approval as a YAML date": _changed("approval.approved_at", datetime.date(2026, 10, 16)),
    "approval blank approver": _changed("approval.approved_by", " "),
    "approval unknown key": _changed("approval.role", "owner"),
    "suite not a mapping": _changed("suite", ["expectations"]),
    "suite unknown key": _changed("suite.meta", {}),
    "suite name mismatch": _changed("suite.name", "synthetic.customers"),
    "no expectations": _changed("suite.expectations", []),
    "expectation not a mapping": _changed("suite.expectations", ["expect_column_values_to_be_unique"]),
    "not allowed": _with("expect_column_mean_to_be_between", {"column": "amount", "min_value": 1}),
    "unknown kwarg": _with("expect_column_values_to_be_unique", {"column": "id", "evil": 1}),
    "forbidden kwarg": _with("expect_column_values_to_be_unique", {"column": "id", "row_condition": "x > 1"}),
    "meta as a kwarg": _with("expect_column_values_to_be_unique", {"column": "id", "meta": {}}),
    "kwargs not a mapping": _with("expect_column_values_to_be_unique", ["id"]),
    "missing regex": _with("expect_column_values_to_match_regex", {"column": "s"}),
    "not-match regex missing": _with("expect_column_values_to_not_match_regex", {"column": "s"}),
    "empty value_set": _with("expect_column_values_to_be_in_set", {"column": "s", "value_set": []}),
    "empty not-in value_set": _with("expect_column_values_to_not_be_in_set", {"column": "s", "value_set": []}),
    "no bound": _with("expect_column_max_to_be_between", {"column": "d"}),
    "length bound strict not bool": _with("expect_column_value_lengths_to_be_between",
                                          {"column": "s", "min_value": 1, "strict_min": "yes"}),
    "pair: empty column": _with("expect_column_pair_values_a_to_be_greater_than_b", {"column_A": "a", "column_B": ""}),
    "pair: bad ignore_row_if": _with("expect_column_pair_values_a_to_be_greater_than_b",
                                     {"column_A": "a", "column_B": "b", "ignore_row_if": "never"}),
    "pair: valid": _with("expect_column_pair_values_a_to_be_greater_than_b",
                         {"column_A": "shipped", "column_B": "ordered", "or_equal": True}),
    "compound: one column": _with("expect_compound_columns_to_be_unique", {"column_list": ["a"]}),
    "compound: repeated column": _with("expect_compound_columns_to_be_unique", {"column_list": ["a", "a"]}),
    "within record: bad ignore_row_if": _with("expect_select_column_values_to_be_unique_within_record",
                                              {"column_list": ["a", "b"], "ignore_row_if": "neither"}),
    "column set empty": _with("expect_table_columns_to_match_set", {"column_set": []}),
    "column set exact_match not bool": _with("expect_table_columns_to_match_set",
                                             {"column_set": ["a"], "exact_match": "no"}),
    "freshness parameter": _with("expect_column_max_to_be_between", {"column": "d", "min_value": _30_DAYS}),
    "stale rows parameter": _with("expect_column_min_to_be_between", {"column": "d", "min_value": _30_DAYS}),
    "free-form parameter": _with("expect_column_max_to_be_between",
                                 {"column": "d", "min_value": {"$PARAMETER": "__import__('os')"}}),
    "parameter with extra keys": _with("expect_column_min_to_be_between",
                                       {"column": "d", "min_value": {**_30_DAYS, "x": 1}}),
    "zero-day parameter": _with("expect_column_max_to_be_between",
                                {"column": "d", "min_value": {"$PARAMETER": "now() - timedelta(days=0)"}}),
    "GX value check": _with("expect_column_values_to_not_be_null", {"column": "a", "mostly": 5}),
    "unknown severity": _with("expect_column_values_to_be_unique", {"column": "id"}, {"severity": "fatal"}),
    "unknown meta key": _with("expect_column_values_to_be_unique", {"column": "id"}, {"owner": "me"}),
    "malformed notes": _with("expect_column_values_to_be_unique", {"column": "id"}, {"notes": "text"}),
    "meta not a mapping": _with("expect_column_values_to_be_unique", {"column": "id"}, "warning"),
    "every expectation, valid": _all_of(
        ("expect_column_values_to_not_be_null", {"column": "a", "mostly": 0.9}),
        ("expect_column_values_to_be_unique", {"column": "a"}),
        ("expect_column_values_to_be_in_set", {"column": "s", "value_set": ["x", "y"]}),
        ("expect_column_values_to_be_between", {"column": "n", "min_value": 0, "max_value": 9}),
        ("expect_column_values_to_match_regex", {"column": "s", "regex": "^[a-z]+$"}),
        ("expect_table_row_count_to_be_between", {"min_value": 1}),
        ("expect_column_values_to_be_in_type_list", {"column": "a", "type_list": ["INTEGER"]}),
        ("expect_column_min_to_be_between", {"column": "d", "min_value": _30_DAYS}),
        ("expect_column_max_to_be_between", {"column": "d", "min_value": _30_DAYS}),
        ("expect_column_values_to_not_be_in_set", {"column": "s", "value_set": ["z"]}),
        ("expect_column_values_to_not_match_regex", {"column": "s", "regex": "^ "}),
        ("expect_column_value_lengths_to_be_between", {"column": "s", "max_value": 8}),
        ("expect_column_pair_values_a_to_be_greater_than_b", {"column_A": "b", "column_B": "a"}),
        ("expect_compound_columns_to_be_unique", {"column_list": ["a", "b"]}),
        ("expect_select_column_values_to_be_unique_within_record",
         {"column_list": ["a", "b"], "ignore_row_if": "any_value_is_missing"}),
        ("expect_table_columns_to_match_set", {"column_set": ["a", "b"], "exact_match": False}),
    ),
}


# What contract_errors must say about each contract: None for a valid one, else a fragment of an error.
_TIMEZONE = "approved_at must include a timezone"
_ISO = "approved_at must be a valid ISO-8601 timestamp"
_PARAMETER = 'the only allowed parameter is {"$PARAMETER": "now() - timedelta(days=N)"}'
VERDICTS = {
    "valid": None,
    "not a mapping": "contract must be a mapping",
    "wrong contract_version": "contract_version must be 1",
    "wrong gx_version": 'gx_version must be "0.18.2"',
    "unknown top-level key": "contract: unknown key 'checkpoint'",
    "dataset not a mapping": "dataset must be a mapping with schema and table",
    "dataset unknown key": "dataset: unknown key 'database'",
    "dataset quoted identifier": "dataset.table must be a plain SQL identifier",
    "no approval": "approval is required",
    "approval date only": _TIMEZONE,
    "approval naive timestamp": _TIMEZONE,
    "approval with offset": None,
    "approval not a timestamp": _ISO,
    "approval as a YAML timestamp": None,
    "approval as a naive YAML timestamp": _TIMEZONE,
    "approval as a YAML date": _ISO,
    "approval blank approver": "approval.approved_by must be a non-empty string",
    "approval unknown key": "approval: unknown key 'role'",
    "suite not a mapping": "suite must be a mapping with name and expectations",
    "suite unknown key": "suite: unknown key 'meta'",
    "suite name mismatch": "suite.name must be 'synthetic.orders'",
    "no expectations": "suite.expectations must be a non-empty list",
    "expectation not a mapping": "expectation 1: must be a mapping",
    "not allowed": "'expect_column_mean_to_be_between' is not an allowed expectation_type",
    "unknown kwarg": "(expect_column_values_to_be_unique on id): unexpected kwarg 'evil' (allowed: column, mostly)",
    "forbidden kwarg": "unexpected kwarg 'row_condition': not allowed in contracts because it changes which rows",
    "meta as a kwarg": "unexpected kwarg 'meta': not allowed in contracts because notes and severity go",
    "kwargs not a mapping": "kwargs must be a mapping",
    "missing regex": "missing required kwarg 'regex'",
    "not-match regex missing": "regex is required",  # GX's own check
    "empty value_set": "value_set must not be empty",
    "empty not-in value_set": "value_set must not be empty",
    "no bound": "needs at least one of min_value, max_value",
    "length bound strict not bool": "strict_min must be true or false",
    "pair: empty column": "column_B must be a non-empty string",
    "pair: bad ignore_row_if": "ignore_row_if must be one of: both_values_are_missing, either_value_is_missing, neither",
    "pair: valid": None,
    "compound: one column": "column_list must list at least two column names",
    "compound: repeated column": "column_list must not repeat a column",
    "within record: bad ignore_row_if": "ignore_row_if must be one of: all_values_are_missing, any_value_is_missing, never",
    "column set empty": "column_set must be a non-empty list of column names",
    "column set exact_match not bool": "exact_match must be true or false",
    "freshness parameter": None,
    "stale rows parameter": None,
    "free-form parameter": _PARAMETER,
    "parameter with extra keys": _PARAMETER,
    "zero-day parameter": _PARAMETER,
    "GX value check": "'mostly' parameter must be between 0 and 1",
    "unknown severity": "meta.severity must be 'error' or 'warning'",
    "unknown meta key": "meta contains unknown key 'owner'",
    "malformed notes": "meta.notes must be {format: string|markdown, content: <text>}",
    "meta not a mapping": "meta must be a mapping",
    "every expectation, valid": None,
}


def test_every_contract_has_a_verdict():
    assert VERDICTS.keys() == CONTRACTS.keys()


@pytest.mark.parametrize("case", CONTRACTS)
def test_verdict(case):
    errors = gx_contract.contract_errors(CONTRACTS[case])
    if VERDICTS[case] is None:
        assert errors == []
    else:
        assert any(VERDICTS[case] in error for error in errors), errors


def test_the_corpus_uses_every_allowed_expectation():
    used = {e["expectation_type"] for e in CONTRACTS["every expectation, valid"]["suite"]["expectations"]}
    assert used == set(gx_contract.ALLOWED_EXPECTATIONS)


def test_errors_name_the_expectation_and_its_columns():
    (error,) = gx_contract.contract_errors(CONTRACTS["compound: repeated column"])
    assert error.startswith("expectation 3 (expect_compound_columns_to_be_unique on a, a): ")


@pytest.mark.parametrize("expectation_type", sorted(gx_contract.ALLOWED_EXPECTATIONS))
def test_allowed_kwargs_come_from_gx_minus_the_forbidden(expectation_type):
    kwargs = gx_contract.allowed_kwargs(expectation_type)
    assert kwargs and not set(kwargs) & gx_contract.FORBIDDEN_KWARGS
    assert len(kwargs) == len(set(kwargs))


def test_policy_judges_any_gx_expectation():
    """kwarg_errors also serves creative mode, whose expectations are not allowed yet."""
    assert gx_contract.kwarg_errors("expect_column_mean_to_be_between", {"column": "a", "min_value": 1}) == []
    assert gx_contract.kwarg_errors("expect_column_mean_to_be_between", {"column": "a"}) == [
        "needs at least one of min_value, max_value"]


# --- loading a contract file ------------------------------------------------
#
# Parametrized over the rules module, so the same tests can be pointed at another copy of the rules.

BOTH = pytest.mark.parametrize("rules", [gx_contract], ids=["gx_contract"])
EXAMPLES = Path(__file__).parent.parent / "contracts" / "examples"


@BOTH
def test_validate_contract_names_the_problem_count(rules):
    with pytest.raises(rules.ContractError, match=r"contract is invalid \(1 problem\)"):
        rules.validate_contract(CONTRACTS["no approval"])
    with pytest.raises(rules.ContractError, match=r"contract is invalid \(2 problems\)"):
        rules.validate_contract(CONTRACTS["approval blank approver"] | {"gx_version": "1.0"})


@BOTH
def test_the_example_contract_loads(rules):
    contract = rules.load_table_contract("synthetic", "orders", EXAMPLES)
    assert contract["suite"]["name"] == "synthetic.orders"


@BOTH
def test_load_table_contract_refuses_a_file_for_another_table(rules, tmp_path):
    (tmp_path / "synthetic.orders.yaml").write_text(
        yaml.safe_dump(_changed("dataset.table", "customers") | {"suite": {
            **VALID["suite"], "name": "synthetic.customers"}}),
        encoding="utf-8",
    )
    with pytest.raises(rules.ContractError, match="not synthetic.orders"):
        rules.load_table_contract("synthetic", "orders", tmp_path)


@BOTH
def test_load_contract_reports_a_missing_file(rules, tmp_path):
    with pytest.raises(rules.ContractError, match="could not be read"):
        rules.load_contract(tmp_path / "missing.yaml")


@BOTH
def test_load_contract_reports_invalid_yaml(rules, tmp_path):
    (tmp_path / "bad.yaml").write_text("suite: [unclosed\n", encoding="utf-8")
    with pytest.raises(rules.ContractError, match="not valid YAML"):
        rules.load_contract(tmp_path / "bad.yaml")


@BOTH
def test_load_contract_names_the_file_when_invalid(rules, tmp_path):
    (tmp_path / "x.yaml").write_text(yaml.safe_dump(CONTRACTS["no approval"]), encoding="utf-8")
    with pytest.raises(rules.ContractError, match=r"x\.yaml is invalid"):
        rules.load_contract(tmp_path / "x.yaml")


@BOTH
@pytest.mark.parametrize("schema, table", [("synthetic", "orders; drop"), ("", "orders"), ("a.b", "orders")])
def test_contract_path_refuses_a_non_identifier(rules, schema, table):
    with pytest.raises(rules.ContractError, match="invalid identifier"):
        rules.contract_path(schema, table, Path("."))


# --- severity on a real GX run ----------------------------------------------

ORDERS = pd.DataFrame({"order_id": [1, 2, 2], "amount": [5.0, -1.0, 3.0]})


def _contract(*expectations: dict) -> dict:
    contract = copy.deepcopy(VALID)
    contract["suite"]["expectations"] = list(expectations)
    return contract


def _expect(expectation_type: str, severity: str | None = None, **kwargs) -> dict:
    entry = {"expectation_type": expectation_type, "kwargs": kwargs}
    if severity:
        entry["meta"] = {"severity": severity}
    return entry


def _run(contract: dict, frame=ORDERS, **options):
    import great_expectations as gx

    context = gx.get_context(mode="ephemeral")
    asset = context.sources.add_pandas("local").add_dataframe_asset("orders")
    return gx_contract.run_contract_on_asset(
        context, contract, asset, checkpoint_name="test", run_name="test",
        batch_request=asset.build_batch_request(dataframe=frame), **options,
    )


UNIQUE = "expect_column_values_to_be_unique"
BETWEEN = "expect_column_values_to_be_between"


def test_a_failed_warning_does_not_fail_the_run(caplog):
    with caplog.at_level(logging.WARNING, logger=gx_contract.__name__):
        result = _run(_contract(_expect(UNIQUE, "warning", column="order_id")))
    assert not result.success  # GX still records the failure
    assert "do not fail the task" in caplog.text


def test_a_failed_error_fails_the_run():
    with pytest.raises(gx_contract.DataQualityCheckFailed, match="1 blocking"):
        _run(_contract(_expect(UNIQUE, "warning", column="order_id"),
                       _expect(BETWEEN, "error", column="amount", min_value=0)))


def test_no_severity_means_error():
    """A contract written before severity existed keeps failing the run."""
    with pytest.raises(gx_contract.DataQualityCheckFailed):
        _run(_contract(_expect(UNIQUE, column="order_id")))


def test_a_check_that_could_not_run_is_an_error_even_as_a_warning():
    result = _run(_contract(_expect(UNIQUE, "warning", column="no_such_column")), raise_on_failure=False)
    (failed,) = gx_contract.failed_expectations(result, "error")
    assert gx_contract.could_not_run(failed)
    assert gx_contract.failed_expectations(result, "warning") == []


MIXED = (
    _expect(UNIQUE, "warning", column="order_id"),
    _expect(BETWEEN, column="amount", min_value=0),
    _expect(UNIQUE, "warning", column="no_such_column"),
    _expect("expect_table_row_count_to_be_between", "warning", min_value=1),
    _expect("expect_compound_columns_to_be_unique", "warning", column_list=["order_id", "amount"]),
)


def test_summary_reports_severity_columns_and_counts_without_values():
    result = _run(_contract(*MIXED), raise_on_failure=False)
    assert {r.expectation_config.kwargs.get("column") for r in gx_contract.failed_expectations(result, "error")} == {
        "amount", "no_such_column"}
    summary = gx_contract.summarize(result)
    assert "FAIL [WARNING] expect_column_values_to_be_unique on order_id (2 unexpected)" in summary
    assert "on no_such_column (could not run)" in summary
    assert "PASS [WARNING] expect_table_row_count_to_be_between on <table>" in summary
    assert "on order_id, amount" in summary
    assert "-1.0" not in summary  # no row values


def test_each_expectation_carries_the_approval():
    contract = _contract(_expect(UNIQUE, "warning", column="order_id"))
    configs = gx_contract.expectation_configurations(contract)
    assert configs[0].meta == {"severity": "warning", "dq_contract": {
        "contract_version": 1, "approved_by": "jacopo", "approved_at": "2026-10-16T09:00:00Z"}}


@pytest.mark.parametrize("newest_days_ago, success", [(0, True), (60, False)])
def test_freshness_is_judged_on_the_newest_row(newest_days_ago, success):
    """The freshness form fails only when the newest date is too old: an old row elsewhere is fine."""
    newest = datetime.date.today() - datetime.timedelta(days=newest_days_ago)
    dates = pd.DataFrame({"d": pd.to_datetime(["2000-01-01", str(newest)])})
    result = _run(_contract(_expect("expect_column_max_to_be_between", column="d", min_value=_30_DAYS)),
                  frame=dates, raise_on_failure=False)
    assert result.success is success


def test_stale_rows_are_judged_on_the_oldest_row():
    dates = pd.DataFrame({"d": pd.to_datetime(["2000-01-01", str(datetime.date.today())])})
    result = _run(_contract(_expect("expect_column_min_to_be_between", column="d", min_value=_30_DAYS)),
                  frame=dates, raise_on_failure=False)
    assert not result.success  # the 2000-01-01 row is stale, though the newest row is fresh


# --- both run_contracts through GX's SQL engine -------------------------------
#
# run_contract targets Postgres; here it is pointed at SQLite (add_postgres -> add_sqlite) so its
# SQL path runs anywhere. tests/test_postgres.py does the same on a real Postgres.

SQL_ORDERS = pd.DataFrame({
    "order_id": [1, 2, 2], "email": ["a@x.io", "b@x.io", "no-at"], "amount": [5.0, -1.0, 3.0],
})


class _SqliteSources:
    def __init__(self, sources):
        self._sources = sources

    def __getattr__(self, name):
        return getattr(self._sources, name)

    def add_postgres(self, name, connection_string):
        return self._sources.add_sqlite(name=name, connection_string=connection_string)

    def add_or_update_postgres(self, name, connection_string):
        return self._sources.add_or_update_sqlite(name=name, connection_string=connection_string)


class _SqliteContext:
    def __init__(self, context):
        self._context = context
        self.sources = _SqliteSources(context.sources)

    def __getattr__(self, name):
        return getattr(self._context, name)


@pytest.fixture
def sqlite_uri(tmp_path):
    path = tmp_path / "orders.db"
    connection = sqlite3.connect(path)
    SQL_ORDERS.to_sql("orders", connection, index=False)
    connection.close()
    return f"sqlite:///{path.as_posix()}"


def _sql_contract(*expectations):
    contract = _contract(*expectations)
    contract["dataset"] = {"schema": "main", "table": "orders"}
    contract["suite"]["name"] = "main.orders"
    return contract


def _sql_run(rules, uri, contract, context=None, **options):
    import great_expectations as gx

    return rules.run_contract(
        context or _SqliteContext(gx.get_context(mode="ephemeral")), contract,
        connection_string=uri, checkpoint_name="test", run_name="test", **options,
    )


SQL_CONTRACT = (
    _expect(UNIQUE, "warning", column="order_id"),
    _expect(BETWEEN, "error", column="amount", min_value=0),
    _expect("expect_compound_columns_to_be_unique", "error", column_list=["order_id", "email"]),
    _expect("expect_table_row_count_to_be_between", "warning", min_value=1, max_value=10),
)


@BOTH
def test_run_contract_raises_on_an_error_failure_over_sql(rules, sqlite_uri):
    with pytest.raises(rules.DataQualityCheckFailed, match="1 blocking data quality check"):
        _sql_run(rules, sqlite_uri, _sql_contract(*SQL_CONTRACT))


@BOTH
def test_run_contract_returns_the_result_when_told_not_to_raise(rules, sqlite_uri, caplog):
    with caplog.at_level(logging.WARNING):
        result = _sql_run(rules, sqlite_uri, _sql_contract(*SQL_CONTRACT), raise_on_failure=False)
    assert [r.expectation_config.expectation_type for r in rules.failed_expectations(result, "error")] == [BETWEEN]
    assert "1 warning-level data quality check" in caplog.text


@BOTH
def test_run_contract_passes_a_clean_contract_over_sql(rules, sqlite_uri):
    result = _sql_run(rules, sqlite_uri, _sql_contract(_expect(BETWEEN, column="amount", min_value=-5)))
    assert result.success


@BOTH
def test_run_contract_validates_before_running(rules, sqlite_uri):
    with pytest.raises(rules.ContractError):
        _sql_run(rules, sqlite_uri, CONTRACTS["no approval"])


def test_a_sql_run_is_summarized(sqlite_uri):
    result = _sql_run(gx_contract, sqlite_uri, _sql_contract(*SQL_CONTRACT), raise_on_failure=False)
    assert "FAIL [ERROR] expect_column_values_to_be_between on amount (1 unexpected)" in gx_contract.summarize(result)


def test_a_rerun_on_one_context_replaces_the_source(sqlite_uri):
    """The notebook re-runs cells on one context: the source is replaced rather than colliding."""
    import great_expectations as gx

    context = _SqliteContext(gx.get_context(mode="ephemeral"))
    contract = _sql_contract(_expect(BETWEEN, column="amount", min_value=-5))
    for _ in range(2):
        assert _sql_run(gx_contract, sqlite_uri, contract, context=context).success
