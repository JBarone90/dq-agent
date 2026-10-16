from pathlib import Path

import pytest
from pydantic import ValidationError

pytest.importorskip("great_expectations")  # problems() asks GX to check kwarg values

# the loader data-flow applies, in dq-agent's copy of its rules
from dq_agent.gx_contract import validate_contract
from dq_agent.models import Contract, Expectation, Meta, Notes

EXAMPLE = Path(__file__).parent.parent / "contracts" / "examples" / "synthetic.orders.yaml"


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
    ("expect_column_values_to_not_match_regex", {"column": "s"}, "regex is required"),  # GX's own check
    ("expect_column_max_to_be_between", {"column": "d"}, "needs at least one of min_value, max_value"),
    ("expect_column_value_lengths_to_be_between", {"column": "s", "min_value": 2, "strict_min": "yes"}, "strict_min must be true or false"),
    ("expect_column_pair_values_a_to_be_greater_than_b", {"column_A": "a", "column_B": ""}, "column_B must be a non-empty string"),
    ("expect_column_pair_values_a_to_be_greater_than_b", {"column_A": "a", "column_B": "b", "or_equal": 1}, "or_equal must be true or false"),
    ("expect_column_pair_values_a_to_be_greater_than_b", {"column_A": "a", "column_B": "b", "ignore_row_if": "never"}, "ignore_row_if must be one of: both_values_are_missing"),
    ("expect_compound_columns_to_be_unique", {"column_list": ["a"]}, "column_list must list at least two column names"),
    ("expect_compound_columns_to_be_unique", {"column_list": ["a", "a"]}, "column_list must not repeat a column"),
    ("expect_select_column_values_to_be_unique_within_record", {"column_list": ["a", "b"], "ignore_row_if": "neither"}, "ignore_row_if must be one of: all_values_are_missing"),
    ("expect_table_columns_to_match_set", {"column_set": []}, "column_set must be a non-empty list of column names"),
    ("expect_table_columns_to_match_set", {"column_set": ["a"], "exact_match": "no"}, "exact_match must be true or false"),
    ("expect_column_values_to_be_unique", {"column": "id", "meta": {}}, "notes and severity go in the expectation's meta"),
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


def test_new_expectations_accepted():
    c = _draft(
        Expectation(expectation_type="expect_column_values_to_not_be_in_set", kwargs={"column": "status", "value_set": ["N/A"]}),
        Expectation(expectation_type="expect_column_pair_values_a_to_be_greater_than_b",
                    kwargs={"column_A": "shipped_at", "column_B": "ordered_at", "or_equal": True}),
        Expectation(expectation_type="expect_compound_columns_to_be_unique", kwargs={"column_list": ["order_id", "line_no"]}),
        Expectation(expectation_type="expect_table_columns_to_match_set", kwargs={"column_set": ["order_id"], "exact_match": False}),
    )
    assert c.problems() == []


def test_problems_name_the_entry_as_a_person_counts_it():
    c = _draft(Expectation(expectation_type="expect_column_values_to_be_unique", kwargs={"column": "id"}),
               Expectation(expectation_type="expect_column_values_to_be_unique", kwargs={"column": "id", "evil": 1}))
    assert any(p.startswith("expectation 2 (expect_column_values_to_be_unique on id)") for p in c.problems())


# --- severity --------------------------------------------------------------


def test_severity_defaults_to_error():
    """No meta means no severity in the YAML, which data-flow reads as an error."""
    assert Expectation(expectation_type="expect_column_values_to_be_unique", kwargs={"column": "id"}).severity == "error"
    assert Meta().severity == "error"


def test_warning_severity_round_trips_and_data_flow_accepts_it():
    c = _draft(Expectation(expectation_type="expect_column_values_to_be_unique", kwargs={"column": "id"},
                           meta=Meta(severity="warning", notes=Notes(content="duplicates are tolerated"))))
    c.approve("jacopo")
    text = c.to_yaml()
    assert "severity: warning" in text
    assert Contract.from_yaml(text).suite.expectations[0].severity == "warning"
    validate_contract(Contract.from_yaml(text).to_dict())  # does not raise


def test_unknown_severity_rejected():
    with pytest.raises(ValidationError):
        Meta(severity="fatal")


def test_meta_without_notes_is_valid():
    c = _draft(Expectation(expectation_type="expect_column_values_to_be_unique", kwargs={"column": "id"},
                           meta=Meta(severity="warning")))
    assert c.problems() == []


def test_approve_returns_the_stamp():
    c = _draft(Expectation(expectation_type="expect_column_values_to_be_unique", kwargs={"column": "id"}))
    approval = c.approve("jacopo")
    assert approval is c.approval and approval.approved_by == "jacopo"
    assert approval.approved_at.tzinfo is not None
