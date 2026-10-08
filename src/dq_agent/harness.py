"""Evaluation harness: scores a proposed contract against the issues it should cover.

Intended for regression testing of the scoping agent — prompt, model, and registry changes should not silently reduce coverage of known dataset issues.

Usage:
    score = score_contract(proposal, ORDERS_EXPECTED_CHECKS)
    assert score.recall == 1.0, f"missed: {score.missed}"

An expected check is an (expectation_type, column) pair the proposal must contain to cover a known issue; table-level expectations have column=None. The comparison is structural — contracts now run in data-flow under Great Expectations, not locally — so a check counts as covered when the proposal includes it, not when it fails on the data. Checks beyond the expected set are reported as `extra`: they are not wrong, only unexpected, and worth a look when a prompt change makes them grow.
"""

from __future__ import annotations

from pydantic import BaseModel

from dq_agent.models import Contract

# An (expectation_type, column-or-None) key identifying one expected check.
CheckKey = tuple[str, str | None]


class HarnessScore(BaseModel, frozen=True):
    caught: list[CheckKey]  # expected checks the proposal includes
    missed: list[CheckKey]  # expected checks the proposal lacks
    extra: list[CheckKey]   # proposed checks outside the expected set
    recall: float           # caught / expected; 1.0 is perfect coverage


def score_contract(proposal: Contract, expected: frozenset[CheckKey]) -> HarnessScore:
    proposed = {
        (e.expectation_type, e.kwargs.get("column")) for e in proposal.suite.expectations
    }
    caught = sorted(k for k in expected if k in proposed)
    return HarnessScore(
        caught=caught,
        missed=sorted(k for k in expected if k not in proposed),
        extra=sorted((k for k in proposed if k not in expected), key=str),
        recall=len(caught) / len(expected) if expected else 1.0,
    )
