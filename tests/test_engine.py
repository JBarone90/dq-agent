import polars as pl
import pytest

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


def test_run_empty_contract_returns_empty_list(orders_df, registry):
    contract = _contract()
    results = run(contract, orders_df, registry)
    assert results == []


def test_run_passes_on_clean_column(orders_df, registry):
    contract = _contract(
        ContractRule(rule_id="null_check", params={"column": "order_id", "max_null_rate": 0.0}),
    )
    result = run(contract, orders_df, registry)[0]
    assert result.passed is True
    assert result.violation_rate == 0.0
