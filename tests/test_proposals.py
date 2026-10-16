"""Creative mode: discovering unlisted GX expectations and writing proposals."""

import pytest
import yaml

pytest.importorskip("great_expectations")  # the catalogue and contract checks are read from GX

from dq_agent.gx_contract import ALLOWED_EXPECTATIONS
from dq_agent import proposals

MEAN = "expect_column_mean_to_be_between"


def test_unlisted_is_gx_minus_the_allowlist():
    unlisted = {u.expectation_type for u in proposals.unlisted_expectations()}
    assert MEAN in unlisted
    assert not unlisted & set(ALLOWED_EXPECTATIONS)
    assert len(unlisted) + len(ALLOWED_EXPECTATIONS) == 53  # GX 0.18.2 core expectations


def test_sql_support_read_from_gx_metric_providers():
    support = {u.expectation_type: u.sql_support for u in proposals.unlisted_expectations()}
    assert support[MEAN] == "supported"
    # strftime parsing has only pandas / spark implementations in 0.18.2
    assert support["expect_column_values_to_match_strftime_format"] == "unverified"


@pytest.mark.parametrize("expectation_type, kwargs, message", [
    ("expect_column_values_to_be_unique", {"column": "id"}, "already allowed"),
    ("expect_column_value_lengths_to_be_between", {"column": "postcode", "min_value": 6}, "already allowed"),
    ("expect_made_up_thing", {}, "not a Great Expectations 0.18.2 expectation"),
    (MEAN, {"column": "amount", "min_value": 10, "row_condition": "x"}, "unexpected kwarg 'row_condition'"),
    (MEAN, {"column": "amount"}, "needs at least one of min_value, max_value"),
])
def test_proposal_errors(expectation_type, kwargs, message):
    assert any(message in e for e in proposals.proposal_errors(expectation_type, kwargs))


def test_valid_proposal_has_no_errors():
    assert proposals.proposal_errors(MEAN, {"column": "amount", "min_value": 10, "max_value": 200}) == []


def _proposal():
    return proposals.new_proposal(
        dataset="synthetic.orders",
        expectation_type=MEAN,
        rationale="the average order is between 10 and 200",
        example_kwargs={"column": "amount", "min_value": 10, "max_value": 200},
        example_pass="a day of typical orders",
        example_fail="a day where a pricing bug zeroes most amounts",
    )


def test_write_proposal_never_overwrites(tmp_path):
    first = proposals.write_proposal(_proposal(), tmp_path)
    second = proposals.write_proposal(_proposal(), tmp_path)
    assert first != second
    saved = yaml.safe_load(first.read_text(encoding="utf-8"))
    assert saved["status"] == "proposed"
    assert saved["sql_support"] == "supported"
    assert saved["gx_description"].startswith("Expect the column mean to be between")
