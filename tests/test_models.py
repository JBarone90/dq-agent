import pytest
from pydantic import ValidationError

from dq_agent.models import Contract, ContractRule, RuleResult


def test_rule_result_passed():
    r = RuleResult(rule_id="null_check", passed=True, metric=0.01)
    assert r.passed is True
    assert r.error is None


def test_rule_result_failed():
    r = RuleResult(rule_id="null_check", passed=False, metric=0.1)
    assert r.passed is False


def test_rule_result_with_error():
    r = RuleResult(rule_id="null_check", passed=False, metric=0.1, error="column missing")
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
