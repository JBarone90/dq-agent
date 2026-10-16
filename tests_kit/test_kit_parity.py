"""The data-flow kit and dq-agent's copy of its rules agree. Home checkout only.

`dataflow_kit/` is a separate component that lives in the data-flow repo; this checkout holds a
copy of it, the work repo does not. So this module sits outside `tests/` (which never mentions
the kit) and runs only when asked for: `uv run pytest tests_kit`. It is what keeps `gx_contract`
honest: same allowlist, same kwargs, and the same verdict, word for word, on every contract in
`tests/test_gx_contract.py`. A difference means a contract the agent approves could be refused in
data-flow (or the other way round): change both sides together.
"""

import os

import pytest

os.environ.setdefault("GE_USAGE_STATS", "FALSE")  # no internet calls from GX, as in tests/conftest.py
pytest.importorskip("great_expectations")
kit = pytest.importorskip("dataflow_kit.operators.dq_validation")

from dq_agent import gx_contract
from tests import test_gx_contract as copy_tests
from tests.test_gx_contract import CONTRACTS, MIXED, SQL_CONTRACT, _contract, _run, _sql_contract, _sql_run

BOTH = pytest.mark.parametrize("rules", [kit, gx_contract], ids=["kit", "gx_contract"])


def test_same_allowlist_and_forbidden_kwargs():
    assert gx_contract.ALLOWED_EXPECTATIONS == kit.ALLOWED_EXPECTATIONS
    assert gx_contract.FORBIDDEN_KWARGS == kit.FORBIDDEN_KWARGS
    assert (gx_contract.CONTRACT_VERSION, gx_contract.GX_VERSION) == (kit.CONTRACT_VERSION, kit.GX_VERSION)


@pytest.mark.parametrize("expectation_type", sorted(kit.ALLOWED_EXPECTATIONS))
def test_same_kwargs_per_expectation(expectation_type):
    assert gx_contract.allowed_kwargs(expectation_type) == kit.allowed_kwargs(expectation_type)


@pytest.mark.parametrize("case", CONTRACTS)
def test_same_verdict_on_every_contract(case):
    assert gx_contract.contract_errors(CONTRACTS[case]) == kit.contract_errors(CONTRACTS[case])


def test_same_example_contract():
    """The kit's example contract and the agent's are the same contract."""
    kit_example = kit.load_table_contract("synthetic", "orders")
    assert kit_example == gx_contract.load_table_contract("synthetic", "orders", copy_tests.EXAMPLES)


@pytest.mark.parametrize("name", [
    "test_validate_contract_names_the_problem_count",
    "test_the_example_contract_loads",
    "test_load_table_contract_refuses_a_file_for_another_table",
    "test_load_contract_reports_a_missing_file",
    "test_load_contract_reports_invalid_yaml",
    "test_load_contract_names_the_file_when_invalid",
])
def test_kit_loads_files_the_same_way(name, tmp_path):
    """The file-loading tests of the copy, rerun with the kit as the rules module."""
    test = getattr(copy_tests, name)
    test(kit, tmp_path=tmp_path) if "tmp_path" in test.__code__.co_varnames else test(kit)


def test_kit_reads_a_result_the_same_way():
    result = _run(_contract(*MIXED), raise_on_failure=False)
    for severity in gx_contract.SEVERITIES:
        assert gx_contract.failed_expectations(result, severity) == kit.failed_expectations(result, severity)
    assert gx_contract.summarize(result) == kit.summarize(result)


def test_same_expectation_configurations():
    contract = _contract(*MIXED)
    assert [c.to_json_dict() for c in gx_contract.expectation_configurations(contract)] == [
        c.to_json_dict() for c in kit.expectation_configurations(contract)]


@BOTH
def test_run_contract_over_sql(rules, sqlite_uri):
    with pytest.raises(rules.DataQualityCheckFailed, match="1 blocking data quality check"):
        _sql_run(rules, sqlite_uri, _sql_contract(*SQL_CONTRACT))


def test_same_summary_of_a_sql_run(sqlite_uri):
    summaries = {rules.__name__: rules.summarize(_sql_run(rules, sqlite_uri, _sql_contract(*SQL_CONTRACT),
                                                          raise_on_failure=False))
                 for rules in (kit, gx_contract)}
    assert len(set(summaries.values())) == 1, summaries


sqlite_uri = copy_tests.sqlite_uri  # the fixture, shared
