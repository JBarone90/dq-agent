from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from dq_agent.models import Contract, ContractRule
from dq_agent.report import render

RUN_AT = datetime(2026, 6, 15, 14, 32, tzinfo=timezone.utc)
APPROVED_AT = datetime(2026, 6, 12, tzinfo=timezone.utc)


def _contract(*rules: ContractRule) -> Contract:
    return Contract(dataset="orders", approved_at=APPROVED_AT, rules=list(rules))


def test_render_header_contains_dataset_and_run_time(registry, orders_df):
    contract = _contract(ContractRule(rule_id="null_check", params={"column": "order_id"}))
    from dq_agent.engine import run

    results = run(contract, orders_df, registry)
    report = render(results, contract, registry, run_at=RUN_AT)

    assert "Dataset: orders" in report
    assert "2026-06-15 14:32 UTC" in report


def test_render_empty_contract(registry, orders_df):
    contract = _contract()
    report = render([], contract, registry, run_at=RUN_AT)
    assert "No rules in contract" in report


def test_render_all_passed_summary(registry, orders_df):
    from dq_agent.engine import run

    contract = _contract(
        ContractRule(rule_id="null_check", params={"column": "order_id"}),
        ContractRule(rule_id="unique_check", params={"column": "email"}),
    )
    results = run(contract, orders_df, registry)
    report = render(results, contract, registry, run_at=RUN_AT)

    assert "All 2 rules passed" in report
    assert "PASSED" in report
    assert "FAILED" not in report


def test_render_all_failed_summary(registry, orders_df):
    from dq_agent.engine import run

    contract = _contract(
        ContractRule(rule_id="null_check", params={"column": "customer_id", "max_null_rate": 0.0}),
        ContractRule(rule_id="unique_check", params={"column": "order_id"}),
    )
    results = run(contract, orders_df, registry)
    report = render(results, contract, registry, run_at=RUN_AT)

    assert "All 2 rules failed" in report
    assert "FAILED" in report
    assert "PASSED" not in report


def test_render_mixed_summary(registry, orders_df):
    from dq_agent.engine import run

    contract = _contract(
        ContractRule(rule_id="null_check", params={"column": "order_id"}),         # passes
        ContractRule(rule_id="null_check", params={"column": "customer_id"}),       # fails
    )
    results = run(contract, orders_df, registry)
    report = render(results, contract, registry, run_at=RUN_AT)

    assert "1 of 2 rules passed, 1 failed" in report
    assert "PASSED" in report
    assert "FAILED" in report


def test_render_shows_rule_name_not_id(registry, orders_df):
    from dq_agent.engine import run

    contract = _contract(ContractRule(rule_id="null_check", params={"column": "order_id"}))
    results = run(contract, orders_df, registry)
    report = render(results, contract, registry, run_at=RUN_AT)

    assert "Null Check" in report
    assert "null_check" not in report


def test_render_shows_column_in_context(registry, orders_df):
    from dq_agent.engine import run

    contract = _contract(ContractRule(rule_id="null_check", params={"column": "customer_id"}))
    results = run(contract, orders_df, registry)
    report = render(results, contract, registry, run_at=RUN_AT)

    assert "customer_id" in report


def test_render_shows_violation_rate(registry, orders_df):
    from dq_agent.engine import run
    from tests.conftest import NULL_CUSTOMER_IDS, TOTAL_ROWS

    contract = _contract(
        ContractRule(rule_id="null_check", params={"column": "customer_id", "max_null_rate": 0.0})
    )
    results = run(contract, orders_df, registry)
    report = render(results, contract, registry, run_at=RUN_AT)

    expected_pct = f"{(NULL_CUSTOMER_IDS / TOTAL_ROWS) * 100:.1f}%"
    assert expected_pct in report


def test_render_shows_error_message(registry, orders_df):
    from dq_agent.engine import run

    contract = _contract(ContractRule(rule_id="null_check", params={}))  # missing column
    results = run(contract, orders_df, registry)
    report = render(results, contract, registry, run_at=RUN_AT)

    assert "error:" in report


def test_render_warning_severity_tag_shown(registry, orders_df):
    """warning severity is shown explicitly — it signals the rule is advisory, not blocking."""
    from dq_agent.engine import run

    contract = _contract(
        ContractRule(rule_id="null_check", params={"column": "customer_id"}, severity="warning")
    )
    results = run(contract, orders_df, registry)
    report = render(results, contract, registry, run_at=RUN_AT)

    assert "[warning]" in report


def test_render_error_severity_not_tagged(registry, orders_df):
    """error is the default severity and adds no tag — failures are critical by default."""
    from dq_agent.engine import run

    contract = _contract(
        ContractRule(rule_id="null_check", params={"column": "customer_id"})
    )
    results = run(contract, orders_df, registry)
    report = render(results, contract, registry, run_at=RUN_AT)

    assert "[error]" not in report


def test_render_min_row_count_shows_threshold_not_column(registry, orders_df):
    from dq_agent.engine import run

    contract = _contract(ContractRule(rule_id="min_row_count", params={"min_rows": 5}))
    results = run(contract, orders_df, registry)
    report = render(results, contract, registry, run_at=RUN_AT)

    assert "min 5 rows" in report


def test_render_example_contract_full_run(registry, orders_df):
    """Smoke test: the example contract renders without errors and covers all known issues."""
    from dq_agent.engine import run

    path = Path(__file__).parent.parent / "contracts" / "examples" / "orders.yaml"
    contract = Contract.from_yaml(path)
    results = run(contract, orders_df, registry)
    report = render(results, contract, registry, run_at=RUN_AT)

    assert "Dataset: orders" in report
    assert "PASSED" in report
    assert "FAILED" in report
    # every column with a known issue appears in the report
    for col in ("customer_id", "order_id", "amount", "email", "status", "created_at"):
        assert col in report
