"""Deterministic renderers that make the tool's artifacts legible to non-engineers.

`render()` turns engine results (list[RuleResult]) into a quality report;
`describe_contract()` turns a contract into a plain-English rule summary for the
scoping-time approval gate. Both join against the registry for human-readable rule
names. No LLM involved — each is a deterministic function of its input.
"""

from __future__ import annotations

from datetime import datetime, timezone

from dq_agent.models import Contract, ContractRule, RuleResult
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


def describe_contract(contract: Contract, registry: Registry) -> str:
    """Render a contract as a plain-English bullet list, one line per rule, for a
    non-technical owner to review at the approval gate. Falls back to the registry's
    rule name when a rule has no custom phrasing, so new rules degrade gracefully."""
    # Markdown: a blank line after the header then "- " items renders as a real list.
    # The driver's approval view (CLI or Streamlit chat) renders this description as markdown.
    lines = [f"Data quality contract for '{contract.dataset}' — {len(contract.rules)} "
             f"rule{'s' if len(contract.rules) != 1 else ''}:", ""]
    for rule in contract.rules:
        lines.append("- " + _describe_rule(rule, registry))
    return "\n".join(lines)


def _describe_rule(rule: ContractRule, registry: Registry) -> str:
    p = rule.params
    col = p.get("column")
    rid = rule.rule_id

    if rid == "null_check":
        rate = p.get("max_null_rate") or 0.0
        text = (f"`{col}` must never be empty" if rate == 0
                else f"`{col}` may be at most {rate * 100:.0f}% empty")
    elif rid == "unique_check":
        text = f"`{col}` must be unique — no duplicate values"
    elif rid == "range_check":
        text = f"`{col}` " + _range_clause(p)
    elif rid == "allowed_values":
        values = ", ".join(str(v) for v in p.get("values", []))
        text = f"`{col}` must be one of: {values}"
    elif rid == "regex_match":
        text = f"`{col}` must match the expected text format"
    elif rid == "freshness":
        text = f"`{col}` must be no more than {p.get('max_days')} days old"
    elif rid == "min_row_count":
        text = f"the table must have at least {p.get('min_rows')} rows"
    else:
        try:
            name = registry.get(rid).name
        except KeyError:
            name = rid
        text = name + (f" on `{col}`" if col else "")

    if rule.severity and rule.severity != "error":
        text += f"  ({rule.severity})"
    return text


def _range_clause(params: dict) -> str:
    lo, hi = params.get("min_val"), params.get("max_val")
    if lo is not None and hi is not None:
        return f"must be between {lo} and {hi}"
    if lo is not None:
        return f"must be at least {lo}"
    if hi is not None:
        return f"must be at most {hi}"
    return "must be within the allowed range"


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
