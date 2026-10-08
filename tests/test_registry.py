import pytest

from dataflow_kit.dq_validation import ALLOWED_EXPECTATIONS, FORBIDDEN_KWARGS
from dq_agent.registry import Registry, RuleDefinition


def test_catalogue_is_the_data_flow_allowlist(registry: Registry):
    assert set(registry.rule_ids) == set(ALLOWED_EXPECTATIONS)


def test_kwargs_and_docs_come_from_gx(registry: Registry):
    rule = registry.get("expect_column_values_to_be_in_set")
    assert list(rule.kwargs) == ["column", "value_set", "mostly"]  # GX keys minus forbidden ones
    assert rule.kwargs["value_set"].description == "A set of objects used for comparison."
    assert rule.description == "Expect each column value to be in a given set."


def test_forbidden_kwargs_never_offered(registry: Registry):
    offered = {k for rule_id in registry.rule_ids for k in registry.get(rule_id).kwargs}
    assert not offered & FORBIDDEN_KWARGS


def test_policy_required_kwargs_are_flagged(registry: Registry):
    assert registry.get("expect_column_values_to_match_regex").required == ["regex"]


def test_notes_for_unknown_expectation_rejected(tmp_path):
    notes = tmp_path / "notes.yaml"
    notes.write_text("expect_something_else:\n  tags: [x]\n")
    with pytest.raises(ValueError, match="not in the allowlist"):
        Registry(notes)


def test_registry_get_returns_rule_definition(registry: Registry):
    rule = registry.get("expect_column_values_to_not_be_null")
    assert isinstance(rule, RuleDefinition)
    assert "completeness" in rule.tags
    assert rule.notes


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


def test_summarize_unknown_rule_falls_back_to_name_and_kwargs(registry: Registry):
    assert registry.summarize("mystery", {"column": "x"}) == "mystery (column=x)"
