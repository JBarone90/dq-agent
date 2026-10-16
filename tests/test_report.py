import pytest

pytest.importorskip("great_expectations")  # the catalogue and contract checks are read from GX

from dq_agent.models import Contract, Expectation, Meta, Notes
from dq_agent.report import describe_contract


def _contract(*expectations: Expectation) -> Contract:
    return Contract.draft("synthetic", "orders", list(expectations))


def _e(expectation_type: str, **kwargs) -> Expectation:
    return Expectation(expectation_type=expectation_type, kwargs=kwargs)


UNIQUE = "expect_column_values_to_be_unique"
NOT_NULL = "expect_column_values_to_not_be_null"


def test_describe_contract_header_counts_checks(registry):
    summary = describe_contract(_contract(_e(UNIQUE, column="order_id"),
                                          _e(NOT_NULL, column="email")), registry)
    assert "synthetic.orders" in summary
    assert "2 checks" in summary


def test_describe_contract_single_check_is_singular(registry):
    assert "1 check:" in describe_contract(_contract(_e(UNIQUE, column="order_id")), registry)


def test_describe_contract_uses_markdown_list(registry):
    """The card renders as markdown — a literal bullet char collapses every item onto
    one line, so items must use '- ' list markers."""
    summary = describe_contract(_contract(_e(UNIQUE, column="order_id"),
                                          _e(NOT_NULL, column="email")), registry)
    assert "\n- **error**: `order_id` must be unique" in summary
    assert "\n- **error**: `email` must never be empty" in summary
    assert "•" not in summary


def test_describe_contract_phrases_every_registry_entry(registry):
    contract = _contract(
        _e(NOT_NULL, column="customer_id"),
        _e(UNIQUE, column="order_id"),
        _e("expect_column_values_to_be_between", column="amount", min_value=0, max_value=500),
        _e("expect_column_values_to_be_in_set", column="status", value_set=["pending", "shipped"]),
        _e("expect_column_values_to_match_regex", column="email", regex="^x$"),
        _e("expect_column_min_to_be_between", column="created_at",
           min_value={"$PARAMETER": "now() - timedelta(days=30)"}),
        _e("expect_table_row_count_to_be_between", min_value=10),
        _e("expect_column_values_to_be_in_type_list", column="amount", type_list=["NUMERIC"]),
    )
    summary = describe_contract(contract, registry)

    assert "`customer_id` must never be empty" in summary
    assert "`amount` must be between 0 and 500" in summary
    assert "`status` must be one of: pending, shipped" in summary
    assert "`email` must match the expected text format" in summary
    assert "`created_at` must have no date earlier than 30 days ago" in summary
    assert "row count must be at least 10" in summary
    assert "`amount` must be stored as NUMERIC" in summary
    # the raw regex is not exposed to a non-technical reviewer
    assert "^x$" not in summary


def test_describe_contract_phrases_the_extended_expectations(registry):
    contract = _contract(
        _e("expect_column_max_to_be_between", column="amount", max_value=10000),
        _e("expect_column_values_to_not_be_in_set", column="status", value_set=["N/A", "test"]),
        _e("expect_column_values_to_not_match_regex", column="email", regex=r"^\s"),
        _e("expect_column_value_lengths_to_be_between", column="postcode", min_value=6, max_value=8),
        _e("expect_column_pair_values_a_to_be_greater_than_b", column_A="shipped_at", column_B="ordered_at"),
        _e("expect_compound_columns_to_be_unique", column_list=["order_id", "line_no"]),
        _e("expect_select_column_values_to_be_unique_within_record", column_list=["home", "away"]),
        _e("expect_table_columns_to_match_set", column_set=["order_id", "amount"]),
    )
    summary = describe_contract(contract, registry)

    assert "the largest value in `amount` must be at most 10000" in summary
    assert "`status` must never be one of: N/A, test" in summary
    assert "`email` must not contain a forbidden text pattern" in summary
    assert "`postcode` must be between 6 and 8 characters long" in summary
    assert "`shipped_at` must be greater than `ordered_at` in every row" in summary
    assert "the combination of order_id, line_no must be unique" in summary
    assert "home, away must all hold different values within each row" in summary
    assert "the table must have the columns: order_id, amount" in summary
    assert r"^\s" not in summary


def test_describe_contract_appends_meta_note(registry):
    e = _e(NOT_NULL, column="customer_id")
    e.meta = Meta(notes=Notes(content="every order has a customer"))
    assert "— every order has a customer" in describe_contract(_contract(e), registry)


def _with_severity(expectation: Expectation, severity: str) -> Expectation:
    expectation.meta = Meta(severity=severity)
    return expectation


def test_describe_contract_explains_what_severity_does(registry):
    summary = describe_contract(_contract(_e(UNIQUE, column="order_id")), registry)
    assert "the pipeline run stops" in summary
    assert "the run still completes" in summary


def test_describe_contract_shows_each_checks_severity(registry):
    summary = describe_contract(_contract(
        _e(UNIQUE, column="order_id"),
        _with_severity(_e(NOT_NULL, column="email"), "warning"),
    ), registry)
    assert "- **error**: `order_id` must be unique" in summary
    assert "- **warning**: `email` must never be empty" in summary
    assert "Every check is set to error" not in summary


def test_describe_contract_flags_a_contract_with_no_warning(registry):
    """An all-error contract stops the run on any failure; the owner should confirm that."""
    summary = describe_contract(_contract(_e(UNIQUE, column="order_id"),
                                          _e(NOT_NULL, column="email")), registry)
    assert "**Every check is set to error**" in summary


def test_describe_contract_skips_a_meta_without_notes(registry):
    summary = describe_contract(_contract(_with_severity(_e(NOT_NULL, column="email"), "warning")), registry)
    assert summary.splitlines()[-1] == "- **warning**: `email` must never be empty"
