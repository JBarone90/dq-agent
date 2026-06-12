import pytest
from pydantic import ValidationError

from dq_agent.models import Contract, ContractRule, RuleResult


def test_rule_result_passed():
    r = RuleResult(rule_id="null_check", passed=True, violation_rate=0.01)
    assert r.passed is True
    assert r.error is None


def test_rule_result_failed():
    r = RuleResult(rule_id="null_check", passed=False, violation_rate=0.1)
    assert r.passed is False


def test_rule_result_with_error():
    r = RuleResult(rule_id="null_check", passed=False, violation_rate=0.1, error="column missing")
    assert r.error == "column missing"


def test_rule_result_rejects_missing_fields():
    with pytest.raises(ValidationError):
        RuleResult(rule_id="null_check", passed=True)


def test_contract_parses():
    c = Contract(
        dataset="orders",
        rules=[ContractRule(rule_id="null_check", params={"column": "customer_id", "max_null_rate": 0.0})],
    )
    assert len(c.rules) == 1
    assert c.approved_at is None


def test_contract_rejects_missing_dataset():
    with pytest.raises(ValidationError):
        Contract(rules=[])


def test_contract_approval_fields_default_to_none():
    c = Contract(dataset="orders", rules=[])
    assert c.approved_by is None
    assert c.columns is None


def test_contract_yaml_round_trip():
    c = Contract(
        dataset="orders",
        approved_at="2026-06-12T00:00:00Z",
        approved_by="jacopo",
        columns={"order_id": "Int64", "amount": "Float64"},
        rules=[
            ContractRule(
                rule_id="null_check",
                params={"column": "customer_id", "max_null_rate": 0.0},
                severity="warning",
            )
        ],
    )
    assert Contract.from_yaml(c.to_yaml()) == c


def test_contract_yaml_omits_unset_optionals():
    c = Contract(dataset="orders", rules=[ContractRule(rule_id="null_check", params={})])
    text = c.to_yaml()
    assert "approved_at" not in text
    assert "severity" not in text
