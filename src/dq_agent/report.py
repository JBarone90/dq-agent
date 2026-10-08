"""Deterministic renderer that makes a contract legible to non-engineers.

`describe_contract()` turns a contract into a plain-English rule summary for the scoping-time approval gate. Each line comes from the rule's `summary` template in the registry, so a new expectation ships its own phrasing with no change here. No LLM involved: the approval card must faithfully represent the contract.
"""

from __future__ import annotations

from dq_agent.models import Contract
from dq_agent.registry import Registry


def describe_contract(contract: Contract, registry: Registry) -> str:
    """Render a contract as a markdown bullet list, one line per expectation."""
    # Markdown: a blank line after the header then "- " items renders as a real list.
    # The driver's approval view (CLI or Streamlit chat) renders this description as markdown.
    expectations = contract.suite.expectations
    lines = [f"Data quality contract for '{contract.dataset.name}' — {len(expectations)} "
             f"check{'s' if len(expectations) != 1 else ''}:", ""]
    for e in expectations:
        line = "- " + registry.summarize(e.expectation_type, e.kwargs)
        if e.meta:
            line += f" — {e.meta.notes.content}"
        lines.append(line)
    return "\n".join(lines)
