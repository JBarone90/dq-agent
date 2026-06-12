from pathlib import Path

import polars as pl
import pytest
import yaml

from dq_agent.engine import run
from dq_agent.models import Contract, ContractRule
from dq_agent.registry import Registry
from tests.conftest import (
    DUPLICATE_ORDER_IDS,
    INVALID_STATUSES,
    NULL_CUSTOMER_IDS,
    TOTAL_ROWS,
    VALID_STATUSES,
)


def _contract(*rules: ContractRule) -> Contract:
    return Contract(dataset="orders", rules=list(rules))


def test_run_returns_one_result_per_rule(orders_df, registry):
    contract = _contract(
        ContractRule(rule_id="null_check", params={"column": "order_id"}),
        ContractRule(rule_id="unique_check", params={"column": "order_id"}),
    )
    results = run(contract, orders_df, registry)
    assert len(results) == 2


def test_run_rule_ids_match_contract_order(orders_df, registry):
    contract = _contract(
        ContractRule(rule_id="null_check", params={"column": "order_id"}),
        ContractRule(rule_id="unique_check", params={"column": "order_id"}),
    )
    results = run(contract, orders_df, registry)
    assert results[0].rule_id == "null_check"
    assert results[1].rule_id == "unique_check"


def test_run_detects_null_violation(orders_df, registry):
    contract = _contract(
        ContractRule(rule_id="null_check", params={"column": "customer_id", "max_null_rate": 0.0}),
    )
    result = run(contract, orders_df, registry)[0]
    assert result.passed is False
    assert result.violation_rate == NULL_CUSTOMER_IDS / TOTAL_ROWS


def test_run_detects_uniqueness_violation(orders_df, registry):
    contract = _contract(
        ContractRule(rule_id="unique_check", params={"column": "order_id"}),
    )
    result = run(contract, orders_df, registry)[0]
    assert result.passed is False
    assert result.violation_rate == DUPLICATE_ORDER_IDS / TOTAL_ROWS


def test_run_detects_allowed_values_violation(orders_df, registry):
    contract = _contract(
        ContractRule(rule_id="allowed_values", params={"column": "status", "values": list(VALID_STATUSES)}),
    )
    result = run(contract, orders_df, registry)[0]
    assert result.passed is False
    assert result.violation_rate == INVALID_STATUSES / TOTAL_ROWS


def test_run_missing_required_param_returns_error_result(orders_df, registry):
    contract = _contract(
        ContractRule(rule_id="null_check", params={}),
    )
    result = run(contract, orders_df, registry)[0]
    assert result.passed is False
    assert result.error is not None
    assert result.violation_rate is None
    assert "column" in result.error


def test_run_unknown_rule_returns_error_result(orders_df, registry):
    contract = _contract(
        ContractRule(rule_id="does_not_exist", params={}),
    )
    result = run(contract, orders_df, registry)[0]
    assert result.passed is False
    assert result.error is not None


def test_run_bad_rule_does_not_block_others(orders_df, registry):
    contract = _contract(
        ContractRule(rule_id="null_check", params={}),          # missing column — will error
        ContractRule(rule_id="unique_check", params={"column": "email"}),  # should pass
    )
    results = run(contract, orders_df, registry)
    assert len(results) == 2
    assert results[0].passed is False
    assert results[0].error is not None
    assert results[1].passed is True


def test_run_empty_dataset_fails_every_rule(orders_df, registry):
    # vacuous pass on zero rows is a silent failure mode; the engine refuses up front
    from tests.test_registry import MINIMAL_PARAMS

    contract = _contract(*[
        ContractRule(rule_id=rule_id, params=params)
        for rule_id, params in MINIMAL_PARAMS.items()
    ])
    results = run(contract, orders_df.head(0), registry)
    assert len(results) == len(MINIMAL_PARAMS)
    for result in results:
        assert result.passed is False, f"'{result.rule_id}' passed on an empty dataset"
        assert result.error == "cannot evaluate: dataset is empty"
        assert result.violation_rate is None, "unevaluated rule must not report a measured rate"


def test_run_empty_contract_returns_empty_list(orders_df, registry):
    contract = _contract()
    results = run(contract, orders_df, registry)
    assert results == []


def test_example_contract_loads_and_runs(orders_df, registry):
    # the README worked example and contracts/examples/orders.yaml must stay executable
    path = Path(__file__).parent.parent / "contracts" / "examples" / "orders.yaml"
    contract = Contract.model_validate(yaml.safe_load(path.read_text()))
    assert contract.approved_at is not None

    results = run(contract, orders_df, registry)
    assert len(results) == len(contract.rules)
    assert all(r.error is None for r in results)

    by_id = {r.rule_id: r for r in results}
    assert by_id["min_row_count"].passed is True
    # every baked-in issue in the synthetic dataset is caught
    for rule_id in ("null_check", "unique_check", "range_check",
                    "allowed_values", "regex_match", "freshness"):
        assert by_id[rule_id].passed is False, f"'{rule_id}' missed its known issue"


def test_run_passes_on_clean_column(orders_df, registry):
    contract = _contract(
        ContractRule(rule_id="null_check", params={"column": "order_id", "max_null_rate": 0.0}),
    )
    result = run(contract, orders_df, registry)[0]
    assert result.passed is True
    assert result.violation_rate == 0.0
