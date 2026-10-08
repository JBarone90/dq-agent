import pytest

from dataflow_kit.dq_validation import ALLOWED_EXPECTATIONS
from dq_agent.registry import Registry, RuleDefinition


def test_registry_matches_the_data_flow_allowlist(registry: Registry):
    """The agent's catalogue and data-flow's allowlist must offer exactly the same
    expectations and kwargs — otherwise the agent proposes what data-flow rejects."""
    assert set(registry.rule_ids) == set(ALLOWED_EXPECTATIONS)
    for rule_id in registry.rule_ids:
        spec = ALLOWED_EXPECTATIONS[rule_id]
        kwargs = registry.get(rule_id).kwargs
        assert set(kwargs) == set(spec["required"]) | set(spec["optional"]), rule_id
        assert {k for k, s in kwargs.items() if s.required} == set(spec["required"]), rule_id


def test_registry_get_returns_rule_definition(registry: Registry):
    rule = registry.get("expect_column_values_to_not_be_null")
    assert isinstance(rule, RuleDefinition)
    assert "completeness" in rule.tags


def test_registry_get_raises_for_unknown_rule(registry: Registry):
    with pytest.raises(KeyError, match="not found in registry"):
        registry.get("nonexistent_rule")


def test_summarize_fills_template(registry: Registry):
    text = registry.summarize("expect_column_values_to_be_in_set",
                              {"column": "status", "value_set": ["a", "b"]})
    assert text == "`status` must be one of: a, b"


def test_summarize_mentions_mostly_tolerance(registry: Registry):
    text = registry.summarize("expect_column_values_to_not_be_null",
                              {"column": "phone", "mostly": 0.9})
    assert text == "`phone` must never be empty (for at least 90% of rows)"


def test_summarize_open_bound_and_freshness(registry: Registry):
    assert registry.summarize("expect_column_values_to_be_between",
                              {"column": "amount", "min_value": 0}) \
        == "`amount` must be between 0 and no limit"
    assert registry.summarize(
        "expect_column_min_to_be_between",
        {"column": "created_at", "min_value": {"$PARAMETER": "now() - timedelta(days=30)"}},
    ) == "`created_at` must have no date older than 30 days"


def test_summarize_unknown_rule_falls_back_to_id(registry: Registry):
    assert registry.summarize("mystery", {"column": "x"}) == "mystery"
