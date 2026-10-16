"""The approval-gate protocol between the scoping graph and the drivers that talk to a person.

The graph pauses at `interrupt()` with an `action_requests` payload and resumes with
`{"decisions": [decision]}`, where a decision is `{"type": "approve" | "edit" | "reject", ...}`.
The CLI and the Streamlit app both go through these helpers, so the shape of that exchange
and the words that count as approval are defined once.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Command

APPROVE_WORDS = frozenset({"approve", "approved", "accept", "yes", "y", "ok", "looks good", "go ahead"})


def pending_interrupt(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """The approval-gate payload if the graph is paused at the gate, else None."""
    pending = (result or {}).get("__interrupt__")
    if not pending:
        return None
    item = pending[0] if isinstance(pending, (list, tuple)) else pending
    return getattr(item, "value", item)


def saved_result(graph: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    """A thread's persisted state as a result dict, with `__interrupt__` set when it is paused
    at the gate, or None for a thread with no history.

    A driver reopening a saved thread must start from this: sending a new message to a thread
    paused at the gate would leave the approval tool call unanswered, which the model rejects
    on every later turn."""
    snapshot = graph.get_state(config)
    if not snapshot.values:
        return None
    result = dict(snapshot.values)
    interrupts = [i for task in snapshot.tasks for i in getattr(task, "interrupts", ())]
    if interrupts:
        result["__interrupt__"] = interrupts
    return result


def resume(decision: dict[str, Any]) -> Command:
    """The command that answers the gate with one decision."""
    return Command(resume={"decisions": [decision]})


def decision_from_text(text: str) -> dict[str, Any]:
    """A typed reply as a decision: an approve word approves, anything else is feedback
    for the agent to revise the draft with."""
    text = text.strip()
    if text.lower() in APPROVE_WORDS:
        return {"type": "approve"}
    return {"type": "reject", "message": text}


def unwrap_decision(response: Any) -> dict[str, Any]:
    """The single decision inside a resume payload, or {} if it has none (which the graph
    treats as not approved).

    Besides `{"decisions": [decision]}`, a bare list, a bare decision dict and a typed
    string are accepted, so a driver that resumes with what the owner typed gets the same
    result as one that goes through `decision_from_text`."""
    if isinstance(response, dict) and "decisions" in response:
        response = response["decisions"] or [{}]
    if isinstance(response, list):
        response = response[0] if response else {}
    if isinstance(response, str):
        return decision_from_text(response)
    return response if isinstance(response, dict) else {}
