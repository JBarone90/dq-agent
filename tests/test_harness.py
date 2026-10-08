"""Evaluation harness for the scoping agent.

PURPOSE
-------
The unit tests here verify that `score_contract()` (in src/dq_agent/harness.py)
computes recall correctly.  The integration test at the bottom runs the
*real* scoping agent against orders.csv and asserts that it catches every known
issue.  Together they form the regression suite for any change to the system
prompt, the LLM model, or the rule registry.

GROUND TRUTH
------------
`ORDERS_EXPECTED_CHECKS` is the hand-written answer key for
data/synthetic/orders.csv.  Every intentional dirty row is covered by a
(expectation_type, column) pair.  If you add a new dirty column to orders.csv, add an
entry here.  If you add a new synthetic dataset, create a new `frozenset` in a
new test file following the same pattern.

RUNNING THE UNIT TESTS  (always on, no LLM required)
-----------------------------------------------------
    uv run pytest tests/test_harness.py -v

RUNNING THE INTEGRATION TEST  (opt-in, requires a live LLM)
------------------------------------------------------------
Set DQ_AGENT_MODEL to any LangChain model string and run the integration marker:

    DQ_AGENT_MODEL=eu.anthropic.claude-sonnet-5 uv run pytest -m integration -v

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
3. Create a new `frozenset` of (expectation_type, column) expected checks.
4. Add an integration test that scopes the new dataset and asserts recall.
"""

import pytest

from dq_agent.harness import score_contract
from dq_agent.models import Contract, Expectation

NOT_NULL = "expect_column_values_to_not_be_null"
UNIQUE = "expect_column_values_to_be_unique"

# Ground truth: the check that covers each known issue in data/synthetic/orders.csv.
# A tuple is (expectation_type, column); None column means a table-level check.
ORDERS_EXPECTED_CHECKS = frozenset({
    (NOT_NULL, "customer_id"),                                # 2 nulls in customer_id
    (UNIQUE, "order_id"),                                     # 1 duplicate order_id
    ("expect_column_values_to_be_between", "amount"),         # 1 negative amount
    ("expect_column_values_to_match_regex", "email"),         # 1 malformed email
    ("expect_column_values_to_be_in_set", "status"),          # 1 invalid status value
    ("expect_column_min_to_be_between", "created_at"),        # 1 stale date
})


def _contract(*checks: tuple[str, str | None]) -> Contract:
    return Contract.draft("synthetic", "orders", [
        Expectation(expectation_type=t, kwargs={"column": c} if c else {}) for t, c in checks
    ])


def test_full_coverage_scores_perfect_recall():
    score = score_contract(_contract(*ORDERS_EXPECTED_CHECKS), ORDERS_EXPECTED_CHECKS)
    assert score.recall == 1.0
    assert score.missed == []
    assert score.extra == []


def test_partial_contract_misses_uncovered_issues():
    score = score_contract(
        _contract((NOT_NULL, "customer_id"), (UNIQUE, "order_id")), ORDERS_EXPECTED_CHECKS
    )
    assert score.caught == sorted([(NOT_NULL, "customer_id"), (UNIQUE, "order_id")])
    assert score.recall == pytest.approx(2 / 6)
    assert len(score.missed) == 4


def test_check_on_wrong_column_is_missed_and_extra():
    score = score_contract(_contract((NOT_NULL, "amount")), ORDERS_EXPECTED_CHECKS)
    assert (NOT_NULL, "customer_id") in score.missed
    assert score.extra == [(NOT_NULL, "amount")]


def test_table_level_check_keys_on_none():
    score = score_contract(
        _contract(("expect_table_row_count_to_be_between", None)), ORDERS_EXPECTED_CHECKS
    )
    assert score.extra == [("expect_table_row_count_to_be_between", None)]


# ---------------------------------------------------------------------------
# Integration tests — live LLM, skipped by default
#
# Two complementary tests exercise the same system prompt and model at different
# difficulty levels:
#
#   explicit  — the user front-loads every rule, column, and parameter. Tests
#               that the agent correctly translates clear instructions into a
#               valid contract. A failure here points to a tool-schema or
#               registry problem, not a prompt problem.
#
#   discovery — the user gives only a brief business description and lets the
#               agent reason from the profile. Tests that the system prompt
#               actually guides the agent to find issues on its own. A failure
#               here points to the system prompt or model capability.
#
# Both tests run against the same ORDERS_EXPECTED_CHECKS ground truth and
# assert recall == 1.0, so any prompt or model change that silently degrades
# coverage is caught by at least one of them.
# ---------------------------------------------------------------------------


def _print_conversation(messages) -> None:
    """Print the full agent interaction trace to stdout (visible with pytest -s)."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    print("\n" + "=" * 60)
    print("AGENT INTERACTION TRACE")
    print("=" * 60)
    for msg in messages:
        if isinstance(msg, SystemMessage):
            # system prompt is injected per-turn, not stored in state — skip
            continue
        elif isinstance(msg, HumanMessage):
            print(f"\n[HUMAN]\n{msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.content:
                print(f"\n[AGENT]\n{msg.content}")
            for tc in getattr(msg, "tool_calls", []):
                args_preview = str(tc["args"])[:200]
                print(f"\n[AGENT → TOOL] {tc['name']}({args_preview})")
        elif isinstance(msg, ToolMessage):
            preview = msg.content[:300] if msg.content else "(empty)"
            print(f"\n[TOOL RESULT]\n{preview}")
        else:
            print(f"\n[{type(msg).__name__}]\n{msg.content}")
    print("\n" + "=" * 60)


def _run_scoping_session(
    graph,
    config: dict,
    csv_path: str,
    first_message: str,
    nudge_message: str,
    max_turns: int = 5,
) -> dict:
    """Drive the graph forward until a draft appears or the approval interrupt fires.

    Returns the final result dict. Raises pytest.fail if no draft is produced
    within max_turns. Always prints the full interaction trace."""
    from langchain_core.messages import HumanMessage

    result = graph.invoke({"messages": [HumanMessage(first_message)]}, config)

    for _ in range(max_turns):
        if result.get("draft") or result.get("__interrupt__"):
            break
        result = graph.invoke({"messages": [HumanMessage(nudge_message)]}, config)

    _print_conversation(result["messages"])

    if not result.get("draft") and not result.get("__interrupt__"):
        last = result["messages"][-1].content if result.get("messages") else "(no messages)"
        pytest.fail(
            f"agent did not produce a draft after {max_turns} turns.\n"
            f"Last agent message: {last[:500]}"
        )

    return result


def _score_and_assert(result, label: str) -> None:
    draft = result.get("draft")
    if draft is None:
        pytest.fail(f"[{label}] no draft in state after session completed")

    score = score_contract(Contract.model_validate(draft), ORDERS_EXPECTED_CHECKS)

    print(f"\n[{label}] recall={score.recall:.2f}")
    print(f"  caught: {score.caught}")
    print(f"  missed: {score.missed}")
    print(f"  extra:  {score.extra}")

    assert score.recall == 1.0, (
        f"[{label}] agent missed {len(score.missed)} known issue(s): {score.missed}\n"
        f"caught: {score.caught}\n"
        f"extra:  {score.extra}"
    )


def _integration_graph(registry, tmp_path):
    pytest.importorskip("langgraph")

    import os
    if not os.environ.get("DQ_AGENT_MODEL"):
        pytest.skip("DQ_AGENT_MODEL not set — see module docstring for instructions")

    from langgraph.checkpoint.memory import InMemorySaver
    from dq_agent.agents.scoping import build_graph

    return build_graph(
        registry=registry,
        contracts_dir=tmp_path,
        checkpointer=InMemorySaver(),
    )


@pytest.mark.integration
def test_scoping_agent_explicit_instructions(
    synthetic_data_path, orders_df, registry, tmp_path
):
    """Agent follows detailed, explicit user requirements and achieves full recall.

    The user front-loads every rule, column, and parameter. A failure here points
    to a tool-schema or registry problem, not a system-prompt problem.

    Run with:
        DQ_AGENT_MODEL=eu.anthropic.claude-sonnet-5 uv run pytest -m integration -s -v
    """
    graph = _integration_graph(registry, tmp_path)
    csv_path = str(synthetic_data_path / "orders.csv")

    result = _run_scoping_session(
        graph,
        config={"configurable": {"thread_id": "integration-explicit"}},
        csv_path=csv_path,
        first_message=(
            f"I need to scope {csv_path} for data quality. "
            "Business context: this is our e-commerce orders table. Each row is a customer "
            "order with an ID, customer reference, purchase amount, email, status, creation "
            "date, and phone. Here are my requirements: "
            "order_id and customer_id must never be null; "
            "order_id must be unique; "
            "amount must be non-negative; "
            "email must match a basic email pattern; "
            "status must be one of: pending, shipped, delivered, cancelled; "
            "no created_at date may be older than 365 days. "
            "The contract will run against the Postgres table synthetic.orders. "
            "Please profile the dataset, propose a contract covering all of these, and "
            "request approval immediately once you have a proposal."
        ),
        nudge_message=(
            "I have given you all the context you need. Please proceed and propose "
            "the contract now."
        ),
    )
    _score_and_assert(result, label="explicit")


@pytest.mark.integration
def test_scoping_agent_discovers_issues_from_profile(
    synthetic_data_path, orders_df, registry, tmp_path
):
    """Agent discovers data quality issues from the profile with minimal user guidance.

    The user provides only a brief business description — no rules, no columns, no
    parameters. The agent must read the profile statistics and reason about what
    checks make sense. A failure here points to the system prompt or model capability.

    Run with:
        DQ_AGENT_MODEL=eu.anthropic.claude-sonnet-5 uv run pytest -m integration -s -v
    """
    graph = _integration_graph(registry, tmp_path)
    csv_path = str(synthetic_data_path / "orders.csv")

    result = _run_scoping_session(
        graph,
        config={"configurable": {"thread_id": "integration-discovery"}},
        csv_path=csv_path,
        first_message=(
            f"I need to scope {csv_path} for data quality. "
            "It's our e-commerce orders table — each row is a customer purchase. "
            "It will run against the Postgres table synthetic.orders. "
            "Please profile it and propose whatever checks make sense given what you find."
        ),
        nudge_message=(
            "Please use your best judgment from the profile and propose a contract now. "
            "Don't wait for more input from me."
        ),
    )
    _score_and_assert(result, label="discovery")
