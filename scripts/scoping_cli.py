"""Terminal driver for the scoping agent — the air-gapped alternative to agent-chat-ui.

This branch has no npm mirror, so there is no agent-chat-ui and no `langgraph dev`
server. This script is the in-process replacement: it compiles the scoping graph,
supplies its own checkpointer, and runs the full conversation + human approval loop
from the terminal. The agent profiles a dataset, proposes a contract, and pauses at
the approval gate; you approve or send feedback, and an approved contract is written
to `contracts/<dataset>.yaml`.

Why a checkpointer at all: the approval gate is a LangGraph `interrupt()`. It pauses
the run by persisting it to the checkpointer (keyed by `thread_id`) and resumes by
looking it back up. With no checkpointer there is nowhere to persist the paused run,
so the gate never completes. `--db` makes the thread durable across restarts; the
default keeps it in memory for the life of the process.

Run (work environment, with the bedrock proxy reachable):
    uv run python scripts/scoping_cli.py
    uv run python scripts/scoping_cli.py --db scoping.sqlite --thread orders-review

Then: point the agent at a dataset — a CSV/Parquet path (e.g. data/synthetic/orders.csv)
or a Postgres table by schema-qualified name (e.g. public.orders, connection from
DATABASE_DSN__datasets_1) — describe its business context, and iterate. Type 'quit' at
any prompt to exit.
"""

from __future__ import annotations

import argparse
import contextlib
import uuid
from typing import Any, Iterator

from langchain_core.messages import AIMessage
from langgraph.types import Command

from dq_agent.agents.bedrock_chat import BedrockProxyError
from dq_agent.agents.scoping import build_graph

_QUIT = {"quit", "exit", ":q"}
_APPROVE = {"approve", "yes", "y", "ok", "looks good"}


def _pending_interrupt(result: dict[str, Any]) -> dict[str, Any] | None:
    """Return the interrupt payload if the graph paused at the approval gate, else None.

    A paused run surfaces its interrupt under the `__interrupt__` key; the payload is
    the dict the `approval` node passed to `interrupt()`. This is what tells a normal
    conversational turn (send a new message) apart from a pending decision (resume)."""
    pending = result.get("__interrupt__")
    if not pending:
        return None
    item = pending[0] if isinstance(pending, (list, tuple)) else pending
    return getattr(item, "value", item)


def _last_ai_text(result: dict[str, Any]) -> str:
    """The latest assistant message text, skipping tool-call-only messages."""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _decision_from_input(text: str) -> dict[str, Any]:
    """Map a typed approval response to a resume decision. Anything that is not an
    explicit approval becomes reject-with-feedback, so the agent can revise the draft
    instead of the session ending."""
    if text.strip().lower() in _APPROVE:
        return {"type": "approve"}
    return {"type": "reject", "message": text.strip()}


def _prompt(label: str) -> str:
    """Read a line; treat EOF (Ctrl-D / Ctrl-Z) and quit words as a request to stop."""
    try:
        text = input(f"\n{label} ")
    except EOFError:
        raise KeyboardInterrupt from None
    if text.strip().lower() in _QUIT:
        raise KeyboardInterrupt
    return text


def _show_gate(payload: dict[str, Any]) -> None:
    """Render the approval interrupt for the owner to review.

    The gate also advertises an `edit` decision, but the terminal cannot offer
    structured editing of the contract, so this driver does not surface it. To
    change the contract, describe the change in plain text: that routes back to the
    agent as feedback (reject-with-feedback) and it revises and re-proposes. The
    Streamlit app is where direct `edit` belongs (an editable contract field)."""
    requests = payload.get("action_requests") or [{}]
    print("\n" + "=" * 72)
    print("APPROVAL REQUESTED")
    print("=" * 72)
    print(requests[0].get("description", "(no description)"))


def converse(graph: Any, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}

    # Kick off with the first user message.
    result = graph.invoke(
        {"messages": [{"role": "user", "content": _prompt("You:")}]}, config
    )

    while True:
        if result.get("contract_path"):
            print(f"\n✅ Contract approved and saved to: {result['contract_path']}")
            return

        payload = _pending_interrupt(result)
        if payload is not None:
            _show_gate(payload)
            answer = _prompt("Approve? [type 'approve', or describe a change to revise]:")
            resume = {"decisions": [_decision_from_input(answer)]}
            result = graph.invoke(Command(resume=resume), config)
        else:
            text = _last_ai_text(result)
            if text:
                print(f"\nAgent: {text}")
            result = graph.invoke(
                {"messages": [{"role": "user", "content": _prompt("You:")}]}, config
            )


@contextlib.contextmanager
def _checkpointer(db: str | None) -> Iterator[Any]:
    """Yield a checkpointer. In-memory by default; SQLite-backed (durable, resumable
    across restarts) when --db is given. SqliteSaver is imported lazily so the common
    in-memory path needs no extra dependency."""
    if db is None:
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()
        return
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(db) as saver:
        yield saver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        metavar="PATH",
        help="SQLite file for a durable, resumable thread (default: in-memory only)",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="Thread id; reuse the same value with --db to resume a conversation",
    )
    args = parser.parse_args()
    thread_id = args.thread or f"scoping-{uuid.uuid4().hex[:8]}"

    print("dq-agent scoping — terminal driver")
    print(f"thread: {thread_id}" + (f"  (durable: {args.db})" if args.db else ""))
    print("Point the agent at a dataset and describe its context. Type 'quit' to exit.")

    with _checkpointer(args.db) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        try:
            converse(graph, thread_id)
        except BedrockProxyError as exc:
            print(f"\n⚠️  The model call failed.\n\n{exc}")
            if args.db:
                print(f"\nThe thread is saved — resume with "
                      f"--db {args.db} --thread {thread_id} once the proxy is reachable.")
        except KeyboardInterrupt:
            print("\nStopped. " + (
                f"Resume with --db {args.db} --thread {thread_id}"
                if args.db
                else "Pass --db to make a session resumable next time."
            ))


if __name__ == "__main__":
    main()
