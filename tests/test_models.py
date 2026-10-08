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
