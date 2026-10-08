from pathlib import Path

import pytest
from pydantic import ValidationError

from dataflow_kit.dq_validation import validate_contract
from dq_agent.models import Contract, Expectation

EXAMPLE = Path(__file__).parent.parent / "dataflow_kit" / "dq_contracts" / "synthetic.orders.yaml"


def _draft(*expectations: Expectation) -> Contract:
    return Contract.draft("synthetic", "orders", list(expectations))


def test_draft_fills_envelope():
    c = _draft(Expectation(expectation_type="expect_column_values_to_be_unique",
                           kwargs={"column": "order_id"}))
    assert c.suite.name == "synthetic.orders"
    assert c.gx_version == "0.18.2"
    assert c.approval is None
    assert c.problems() == []


def test_problems_come_from_the_data_flow_allowlist():
    c = _draft(Expectation(expectation_type="expect_column_values_to_be_unique",
                           kwargs={"column": "order_id", "evil": 1}))
    assert any("unexpected kwarg 'evil'" in p for p in c.problems())


def test_approved_contract_is_what_data_flow_loads():
    c = _draft(Expectation(expectation_type="expect_column_values_to_be_unique",
                           kwargs={"column": "order_id"}))
    c.approve("jacopo")
    validate_contract(Contract.from_yaml(c.to_yaml()).to_dict())  # does not raise


def test_yaml_round_trip_keeps_schema_alias():
    c = Contract.from_yaml(EXAMPLE)
    text = c.to_yaml()
    assert "schema: synthetic" in text
    assert Contract.from_yaml(text) == c


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        Contract.model_validate({**Contract.from_yaml(EXAMPLE).to_dict(), "checkpoint": {}})


@pytest.mark.parametrize("expectation_type, kwargs, message", [
    ("expect_column_values_to_be_unique", {"column": "id", "row_condition": "x > 1"}, "unexpected kwarg 'row_condition'"),
    ("expect_column_values_to_be_in_set", {"column": "s", "value_set": "a"}, "value_set must be a list"),  # GX's own check
    ("expect_column_values_to_be_in_set", {"column": "s", "value_set": []}, "value_set must not be empty"),
    ("expect_column_values_to_match_regex", {"column": "s"}, "missing required kwarg 'regex'"),
    ("expect_table_row_count_to_be_between", {}, "needs at least one of min_value, max_value"),
    ("expect_column_values_to_not_be_null", {"column": "a", "mostly": 5}, "'mostly' parameter must be between 0 and 1"),
    ("expect_column_min_to_be_between", {"column": "d", "min_value": {"$PARAMETER": "__import__('os')"}}, "the only allowed parameter"),
])
def test_problems_reject(expectation_type, kwargs, message):
    c = _draft(Expectation(expectation_type=expectation_type, kwargs=kwargs))
    assert any(message in p for p in c.problems()), c.problems()


def test_freshness_parameter_accepted():
    c = _draft(Expectation(
        expectation_type="expect_column_min_to_be_between",
        kwargs={"column": "d", "min_value": {"$PARAMETER": "now() - timedelta(days=30)"}},
    ))
    assert c.problems() == []
