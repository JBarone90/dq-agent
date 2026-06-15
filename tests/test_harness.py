"""Evaluation harness for the scoping agent.

PURPOSE
-------
The unit tests here verify that `score_contract()` (in src/dq_agent/harness.py)
computes recall/precision correctly.  The integration test at the bottom runs the
*real* scoping agent against orders.csv and asserts that it catches every known
issue.  Together they form the regression suite for any change to the system
prompt, the LLM model, or the rule registry.

GROUND TRUTH
------------
`ORDERS_EXPECTED_FAILURES` is the hand-written answer key for
data/synthetic/orders.csv.  Every intentional dirty row is listed as a
(rule_id, column) pair.  If you add a new dirty column to orders.csv, add an
entry here.  If you add a new synthetic dataset, create a new `frozenset` in a
new test file following the same pattern.

RUNNING THE UNIT TESTS  (always on, no LLM required)
-----------------------------------------------------
    uv run pytest tests/test_harness.py -v

RUNNING THE INTEGRATION TEST  (opt-in, requires a live LLM)
------------------------------------------------------------
Set DQ_AGENT_MODEL to any LangChain model string and run the integration marker:

    DQ_AGENT_MODEL=google_genai:gemini-2.5-flash uv run pytest -m integration -v

The integration test drives the scoping agent through a full session, extracts
the draft contract it produces, and asserts recall == 1.0 (every known issue
caught).  If the assertion fails the error message lists exactly which issues
were missed — that tells you whether to fix the system prompt, the registry, or
both.

To exclude integration tests from a normal run (the default):

    uv run pytest            # integration tests are skipped automatically
    uv run pytest -m "not integration"   # explicit form of the same thing

ADDING A NEW DATASET
--------------------
1. Create data/synthetic/<name>.csv with known dirty rows.
2. Document the dirty rows in tests/conftest.py (follow the orders pattern).
3. Create a new `frozenset` of (rule_id, column) expected failures.
4. Add a hand-written ideal contract as a sanity check (equivalent to
   `IDEAL_RULES` below).
5. Add an integration test that scopes the new dataset and asserts recall.
"""

import datetime
from datetime import timezone

import pytest

from dq_agent.harness import score_contract
from dq_agent.models import Contract, ContractRule
from tests.conftest import EMAIL_PATTERN, FRESHNESS_THRESHOLD_DAYS, VALID_STATUSES

APPROVED_AT = datetime.datetime(2026, 6, 12, tzinfo=timezone.utc)

# Ground truth: every known issue baked into data/synthetic/orders.csv.
# A tuple is (rule_id, column); None column means a table-level rule.
ORDERS_EXPECTED_FAILURES = frozenset({
    ("null_check", "customer_id"),   # 2 nulls in customer_id
    ("unique_check", "order_id"),    # 1 duplicate order_id
    ("range_check", "amount"),       # 1 negative amount
    ("regex_match", "email"),        # 1 malformed email
    ("allowed_values", "status"),    # 1 invalid status value
    ("freshness", "created_at"),     # 1 stale date
})

IDEAL_RULES = [
    ContractRule(rule_id="null_check", params={"column": "customer_id", "max_null_rate": 0.0}),
    ContractRule(rule_id="unique_check", params={"column": "order_id"}),
    ContractRule(rule_id="range_check", params={"column": "amount", "min_val": 0.0}),
    ContractRule(
        rule_id="regex_match",
        params={"column": "email", "pattern": EMAIL_PATTERN},
    ),
    ContractRule(
        rule_id="allowed_values",
        params={"column": "status", "values": list(VALID_STATUSES)},
    ),
    ContractRule(
        rule_id="freshness",
        params={"column": "created_at", "max_days": FRESHNESS_THRESHOLD_DAYS,
                "as_of": datetime.date(2026, 6, 12)},
    ),
]


def _contract(*rules: ContractRule) -> Contract:
    return Contract(dataset="orders", approved_at=APPROVED_AT, rules=list(rules))


def test_ideal_contract_scores_perfect_recall(orders_df, registry):
    score = score_contract(
        _contract(*IDEAL_RULES), orders_df, registry, ORDERS_EXPECTED_FAILURES
    )
    assert score.recall == 1.0
    assert score.missed == []
    assert score.spurious == []
    assert len(score.caught) == len(ORDERS_EXPECTED_FAILURES)


def test_partial_contract_misses_uncovered_issues(orders_df, registry):
    partial = _contract(
        ContractRule(rule_id="null_check", params={"column": "customer_id", "max_null_rate": 0.0}),
        ContractRule(rule_id="unique_check", params={"column": "order_id"}),
    )
    score = score_contract(partial, orders_df, registry, ORDERS_EXPECTED_FAILURES)

    assert ("null_check", "customer_id") in score.caught
    assert ("unique_check", "order_id") in score.caught
    assert score.recall < 1.0
    assert len(score.missed) == len(ORDERS_EXPECTED_FAILURES) - 2


def test_empty_contract_misses_all_issues(orders_df, registry):
    score = score_contract(_contract(), orders_df, registry, ORDERS_EXPECTED_FAILURES)
    assert score.recall == 0.0
    assert score.caught == []
    assert len(score.missed) == len(ORDERS_EXPECTED_FAILURES)


def test_spurious_rule_detected(orders_df, registry):
    """A rule that fails for an issue not in expected_failures is spurious."""
    # amount has 1 null — null_check on amount fails but is not in expected_failures
    contract = _contract(
        ContractRule(rule_id="null_check", params={"column": "customer_id", "max_null_rate": 0.0}),
        ContractRule(rule_id="null_check", params={"column": "amount", "max_null_rate": 0.0}),
    )
    score = score_contract(contract, orders_df, registry, ORDERS_EXPECTED_FAILURES)

    assert ("null_check", "customer_id") in score.caught
    assert ("null_check", "amount") in score.spurious
    assert score.precision < 1.0


def test_error_results_do_not_count_as_caught(orders_df, registry):
    """Misconfigured rules (missing required param) produce errors, not data catches."""
    contract = _contract(
        ContractRule(rule_id="null_check", params={}),  # missing required 'column'
    )
    score = score_contract(contract, orders_df, registry, ORDERS_EXPECTED_FAILURES)
    assert score.caught == []
    assert score.recall == 0.0


def test_clean_column_rule_not_in_caught_or_spurious(orders_df, registry):
    """A rule that passes is neither caught nor spurious — it's just a passing check."""
    contract = _contract(
        ContractRule(rule_id="null_check", params={"column": "order_id"}),  # no nulls
    )
    score = score_contract(contract, orders_df, registry, ORDERS_EXPECTED_FAILURES)
    assert score.caught == []
    assert score.spurious == []
    assert score.missed == sorted(ORDERS_EXPECTED_FAILURES)


def test_recall_and_precision_values(orders_df, registry):
    """Verify the arithmetic: 2 caught of 6 expected, 0 spurious → recall=1/3, precision=1.0."""
    partial = _contract(
        ContractRule(rule_id="null_check", params={"column": "customer_id", "max_null_rate": 0.0}),
        ContractRule(rule_id="unique_check", params={"column": "order_id"}),
    )
    score = score_contract(partial, orders_df, registry, ORDERS_EXPECTED_FAILURES)

    assert score.recall == pytest.approx(2 / 6)
    assert score.precision == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration test — live LLM, skipped by default
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_scoping_agent_achieves_full_recall_on_orders(
    synthetic_data_path, orders_df, registry, tmp_path
):
    """The real scoping agent must catch every known issue in orders.csv.

    Run with:
        DQ_AGENT_MODEL=google_genai:gemini-2.5-flash uv run pytest -m integration -v

    Failure output lists exactly which issues were missed so you know whether to
    fix the system prompt, the registry, or both.
    """
    import os

    if not os.environ.get("DQ_AGENT_MODEL"):
        pytest.skip("DQ_AGENT_MODEL not set — see module docstring for instructions")

    pytest.importorskip("langgraph")

    from datetime import datetime, timezone

    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from dq_agent.agents.scoping import build_graph
    from dq_agent.models import Contract

    graph = build_graph(
        registry=registry,
        contracts_dir=tmp_path,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "integration-orders"}}
    csv_path = str(synthetic_data_path / "orders.csv")

    result = graph.invoke(
        {"messages": [HumanMessage(
            f"I need to scope {csv_path} for data quality. "
            "Business context: this is our e-commerce orders table. Each row is a customer "
            "order with an ID, customer reference, purchase amount, email, status, creation "
            "date, and phone. Here are my requirements: "
            "order_id and customer_id must never be null (max_null_rate: 0.0); "
            "order_id must be unique; "
            "amount must be non-negative (min_val: 0.0); "
            "email must match a basic email pattern; "
            "status must be one of: pending, shipped, delivered, cancelled; "
            "created_at must be no older than 365 days (use 2026-06-12 as the reference date). "
            "Please profile the dataset, propose a contract covering all of these, and "
            "request approval immediately once you have a proposal."
        )]},
        config,
    )

    # The agent is conversational and may ask a clarifying question before proposing.
    # Nudge it forward up to MAX_TURNS times until a draft appears or the approval
    # interrupt fires.
    MAX_TURNS = 5
    for _ in range(MAX_TURNS):
        if result.get("draft") or result.get("__interrupt__"):
            break
        result = graph.invoke(
            {"messages": [HumanMessage(
                "I have given you all the context you need. Please proceed and propose "
                "the contract now."
            )]},
            config,
        )

    draft = result.get("draft")
    if draft is None:
        last_msg = result["messages"][-1].content if result.get("messages") else "(no messages)"
        pytest.fail(
            f"agent did not produce a draft after {MAX_TURNS} turns.\n"
            f"Last agent message: {last_msg[:500]}"
        )

    contract = Contract.model_validate(draft)
    contract.approved_at = datetime.now(timezone.utc)  # allow engine to run

    score = score_contract(contract, orders_df, registry, ORDERS_EXPECTED_FAILURES)
    assert score.recall == 1.0, (
        f"agent missed {len(score.missed)} known issue(s): {score.missed}\n"
        f"caught:   {score.caught}\n"
        f"spurious: {score.spurious}"
    )
