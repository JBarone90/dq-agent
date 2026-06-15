"""Deterministic renderer: list[RuleResult] → a human-readable quality report.

`render()` is the only entry point. It joins results with the contract (for params)
and the registry (for rule names) so the output makes sense without engineering
knowledge. No LLM involved — the report is a deterministic function of the results.
"""

from __future__ import annotations

from datetime import datetime, timezone

from dq_agent.models import Contract, RuleResult
from dq_agent.registry import Registry


def render(
    results: list[RuleResult],
    contract: Contract,
    registry: Registry,
    *,
    run_at: datetime | None = None,
) -> str:
    if run_at is None:
        run_at = datetime.now(timezone.utc)

    lines: list[str] = []
    lines.append(
        f"Dataset: {contract.dataset}  |  "
        f"Run: {run_at.strftime('%Y-%m-%d %H:%M UTC')}"
    )
    lines.append("")

    if not results:
        lines.append("No rules in contract.")
        return "\n".join(lines)

    n_passed = sum(1 for r in results if r.passed)
    n_failed = len(results) - n_passed

    if n_failed == 0:
        summary = f"All {len(results)} rule{'s' if len(results) != 1 else ''} passed."
    elif n_passed == 0:
        summary = f"All {len(results)} rule{'s' if len(results) != 1 else ''} failed."
    else:
        summary = f"{n_passed} of {len(results)} rules passed, {n_failed} failed."
    lines.append(summary)

    params_for = {cr.rule_id: cr.params for cr in contract.rules}

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    if passed:
        lines.append("")
        lines.append("PASSED")
        for r in passed:
            lines.append(_format_line(r, params_for.get(r.rule_id, {}), registry))

    if failed:
        lines.append("")
        lines.append("FAILED")
        for r in failed:
            lines.append(_format_line(r, params_for.get(r.rule_id, {}), registry))

    return "\n".join(lines)


def _format_line(result: RuleResult, params: dict, registry: Registry) -> str:
    try:
        name = registry.get(result.rule_id).name
    except KeyError:
        name = result.rule_id

    severity_tag = (
        f" [{result.severity}]"
        if result.severity and result.severity != "error"
        else ""
    )
    label = f"  {name}{severity_tag}"
    context = _context_label(params)

    if result.error:
        detail = f"error: {result.error}"
    elif result.violation_rate is not None:
        detail = f"{result.violation_rate * 100:.1f}% violation rate"
    else:
        detail = "not evaluated"

    if context:
        return f"{label:<42} {context:<22} {detail}"
    return f"{label:<42} {detail}"


def _context_label(params: dict) -> str:
    if "column" in params:
        return params["column"]
    if "min_rows" in params:
        return f"min {params['min_rows']} rows"
    return ""
