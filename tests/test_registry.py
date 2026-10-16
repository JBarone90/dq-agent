import pytest

pytest.importorskip("great_expectations")  # the catalogue and contract checks are read from GX

from dq_agent.gx_contract import ALLOWED_EXPECTATIONS, FORBIDDEN_KWARGS
from dq_agent.registry import Registry, RuleDefinition


def test_catalogue_is_the_data_flow_allowlist(registry: Registry):
    assert set(registry.rule_ids) == set(ALLOWED_EXPECTATIONS)


def test_every_allowed_expectation_has_a_summary_and_notes(registry: Registry):
    """The approval card shows the summary to a non-engineer; without one it shows the GX name."""
    for rule_id in registry.rule_ids:
        rule = registry.get(rule_id)
        assert rule.summary and rule.notes and rule.tags, rule_id


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


@pytest.mark.parametrize("expectation_type, kwargs, text", [
    ("expect_column_values_to_be_between", {"column": "amount", "min_value": 0}, "`amount` must be at least 0"),
    ("expect_column_values_to_be_between", {"column": "amount", "max_value": 9}, "`amount` must be at most 9"),
    ("expect_column_values_to_be_between", {"column": "amount", "min_value": 0, "max_value": 9},
     "`amount` must be between 0 and 9"),
    # freshness: the newest date is recent
    ("expect_column_max_to_be_between",
     {"column": "created_at", "min_value": {"$PARAMETER": "now() - timedelta(days=2)"}},
     "the largest value in `created_at` must be no earlier than 2 days ago"),
    # stale rows: no date is old
    ("expect_column_min_to_be_between",
     {"column": "created_at", "min_value": {"$PARAMETER": "now() - timedelta(days=30)"}},
     "`created_at` must have no date earlier than 30 days ago"),
])
def test_summarize_one_sided_bounds_and_dates(registry: Registry, expectation_type, kwargs, text):
    assert registry.summarize(expectation_type, kwargs) == text


def test_summarize_unknown_rule_falls_back_to_name_and_kwargs(registry: Registry):
    assert registry.summarize("mystery", {"column": "x"}) == "mystery (column=x)"


def test_summarize_multi_column_expectations(registry: Registry):
    assert registry.summarize("expect_column_pair_values_a_to_be_greater_than_b",
                              {"column_A": "shipped_at", "column_B": "ordered_at"})         == "`shipped_at` must be greater than `ordered_at` in every row"
    assert registry.summarize("expect_compound_columns_to_be_unique",
                              {"column_list": ["order_id", "line_no"]})         == "the combination of order_id, line_no must be unique"
    assert registry.summarize("expect_table_columns_to_match_set",
                              {"column_set": ["a", "b"]}) == "the table must have the columns: a, b"
