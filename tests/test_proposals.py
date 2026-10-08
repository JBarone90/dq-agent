"""Creative mode: discovering unlisted GX expectations and writing proposals."""

import pytest
import yaml

from dataflow_kit.dq_validation import ALLOWED_EXPECTATIONS
from dq_agent import proposals

LENGTHS = "expect_column_value_lengths_to_be_between"


def test_unlisted_is_gx_minus_the_allowlist():
    unlisted = {u.expectation_type for u in proposals.unlisted_expectations()}
    assert LENGTHS in unlisted
    assert not unlisted & set(ALLOWED_EXPECTATIONS)
    assert len(unlisted) + len(ALLOWED_EXPECTATIONS) == 53  # GX 0.18.2 core expectations


def test_sql_support_read_from_gx_metric_providers():
    support = {u.expectation_type: u.sql_support for u in proposals.unlisted_expectations()}
    assert support[LENGTHS] == "supported"
    # strftime parsing has only pandas / spark implementations in 0.18.2
    assert support["expect_column_values_to_match_strftime_format"] == "unverified"


@pytest.mark.parametrize("expectation_type, kwargs, message", [
    ("expect_column_values_to_be_unique", {"column": "id"}, "already allowed"),
    ("expect_made_up_thing", {}, "not a Great Expectations 0.18.2 expectation"),
    (LENGTHS, {"column": "postcode", "min_value": 6, "row_condition": "x"}, "unexpected kwarg 'row_condition'"),
    (LENGTHS, {"column": "postcode"}, "needs at least one of min_value, max_value"),
])
def test_proposal_errors(expectation_type, kwargs, message):
    assert any(message in e for e in proposals.proposal_errors(expectation_type, kwargs))


def test_valid_proposal_has_no_errors():
    assert proposals.proposal_errors(LENGTHS, {"column": "postcode", "min_value": 6, "max_value": 8}) == []


def _proposal():
    return proposals.new_proposal(
        dataset="synthetic.orders",
        expectation_type=LENGTHS,
        rationale="UK postcodes are 6-8 characters",
        example_kwargs={"column": "postcode", "min_value": 6, "max_value": 8},
        example_pass="SW1A 1AA",
        example_fail="SW1",
    )


def test_write_proposal_never_overwrites(tmp_path):
    first = proposals.write_proposal(_proposal(), tmp_path)
    second = proposals.write_proposal(_proposal(), tmp_path)
    assert first != second
    saved = yaml.safe_load(first.read_text(encoding="utf-8"))
    assert saved["status"] == "proposed"
    assert saved["sql_support"] == "supported"
    assert saved["gx_description"].startswith("Expect the column entries to be strings with length")
