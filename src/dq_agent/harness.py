"""Evaluation harness: scores a proposed contract against a set of expected failures.

Intended for regression testing of the scoping agent — prompt, model, and registry
changes should not silently reduce coverage of known dataset issues.

Usage:
    score = score_contract(proposal, df, registry, ORDERS_EXPECTED_FAILURES)
    assert score.recall == 1.0, f"missed: {score.missed}"

An "expected failure" is a (rule_id, column) pair that should produce a failing,
non-error RuleResult on the target dataset. Table-level rules have column=None.
Rules that error (misconfiguration) do not count as caught — the data issue was
never actually measured.
"""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel

from dq_agent.engine import run
from dq_agent.models import Contract
from dq_agent.registry import Registry

# A (rule_id, column-or-None) key identifying one expected data quality failure.
IssueKey = tuple[str, str | None]


class HarnessScore(BaseModel, frozen=True):
    caught: list[IssueKey]    # expected issues the proposal correctly detected
    missed: list[IssueKey]    # expected issues the proposal failed to detect
    spurious: list[IssueKey]  # rules that failed for issues not in expected_failures
    recall: float             # caught / (caught + missed); 1.0 is perfect coverage
    precision: float          # caught / (caught + spurious); 1.0 means no noise


def score_contract(
    proposal: Contract,
    df: pl.DataFrame,
    registry: Registry,
    expected_failures: frozenset[IssueKey],
) -> HarnessScore:
    results = run(proposal, df, registry)

    failed_issues: set[IssueKey] = set()
    for result, contract_rule in zip(results, proposal.rules):
        # errors mean the rule never ran — misconfiguration, not a data catch
        if not result.passed and result.error is None:
            key: IssueKey = (result.rule_id, contract_rule.params.get("column"))
            failed_issues.add(key)

    caught = sorted(k for k in expected_failures if k in failed_issues)
    missed = sorted(k for k in expected_failures if k not in failed_issues)
    spurious = sorted(k for k in failed_issues if k not in expected_failures)

    recall = len(caught) / len(expected_failures) if expected_failures else 1.0
    precision = len(caught) / len(failed_issues) if failed_issues else 1.0

    return HarnessScore(
        caught=caught,
        missed=missed,
        spurious=spurious,
        recall=recall,
        precision=precision,
    )
