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
    assert "\n- `order_id` must be unique" in summary
    assert "\n- `email` must never be empty" in summary
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
    assert "`created_at` must have no date older than 30 days" in summary
    assert "row count must be between 10 and no limit" in summary
    assert "`amount` must be stored as NUMERIC" in summary
    # the raw regex is not exposed to a non-technical reviewer
    assert "^x$" not in summary


def test_describe_contract_appends_meta_note(registry):
    e = _e(NOT_NULL, column="customer_id")
    e.meta = Meta(notes=Notes(content="every order has a customer"))
    assert "— every order has a customer" in describe_contract(_contract(e), registry)
