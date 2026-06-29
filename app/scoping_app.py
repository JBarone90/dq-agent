"""Streamlit chat interface for the scoping agent — the non-coder-facing driver.

This is the visual counterpart to `scripts/scoping_cli.py`: same in-process graph,
same checkpointer requirement, same interrupt/resume contract — just rendered as a
chat panel a dataset owner can use without touching a terminal. It deliberately
mirrors the parts of agent-chat-ui that matter here:

  - a chat transcript with the agent,
  - **visible tool calls** (profile_dataset / list_rules / propose_contract) with a
    sidebar **toggle** to show or hide them,
  - the human approval gate rendered as approve / edit / reject controls, where
    *edit* lets the owner change the contract YAML before approving.

Run from the repo root (needs the work environment for the Bedrock proxy to answer):

    uv run --extra ui streamlit run app/scoping_app.py

Scaffold status: the chat + approval loop is wired against the real graph. Two
sidebar features are stubs pending an adapter change — token usage reads
`AIMessage.usage_metadata`, which `DeptBedrockChat` does not yet populate, and
streaming is not available because the adapter has no `_stream` (see the
DeptBedrockChat limitations docstring).
"""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st
import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from dq_agent.agents.scoping import build_graph
from dq_agent.models import Contract

st.set_page_config(page_title="dq-agent · scoping", page_icon="🧪", layout="centered")


def _init_state() -> None:
    """Build the graph once per session and hold it (with its checkpointer) in
    session_state, so reruns reuse the same thread instead of resetting it."""
    if "graph" not in st.session_state:
        st.session_state.checkpointer = MemorySaver()
        st.session_state.graph = build_graph(checkpointer=st.session_state.checkpointer)
        st.session_state.thread_id = f"scoping-{uuid.uuid4().hex[:8]}"
        st.session_state.result = None  # latest graph result; holds the full transcript


def _config() -> dict[str, Any]:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _pending_interrupt(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """The approval-gate payload if the graph is paused, else None — the same
    __interrupt__ check the CLI uses to tell a turn apart from a resume."""
    pending = (result or {}).get("__interrupt__")
    if not pending:
        return None
    item = pending[0] if isinstance(pending, (list, tuple)) else pending
    return getattr(item, "value", item)


def _token_usage(result: dict[str, Any] | None) -> tuple[int, int]:
    """Sum input/output tokens across the transcript. Returns (0, 0) until the
    adapter populates AIMessage.usage_metadata."""
    inp = out = 0
    for msg in (result or {}).get("messages", []):
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            inp += usage.get("input_tokens", 0)
            out += usage.get("output_tokens", 0)
    return inp, out


def _render_transcript(result: dict[str, Any] | None, show_tools: bool) -> None:
    """Render the conversation. Tool calls and tool results are gated behind the
    show_tools toggle so a non-technical owner sees a clean chat by default but can
    open the hood to see what the agent actually did."""
    for msg in (result or {}).get("messages", []):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            if msg.content:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)
            if show_tools and msg.tool_calls:
                with st.chat_message("assistant"):
                    for call in msg.tool_calls:
                        with st.expander(f"🔧 tool call · {call['name']}", expanded=False):
                            st.json(call["args"])
        elif isinstance(msg, ToolMessage) and show_tools:
            label = msg.name or msg.tool_call_id
            with st.chat_message("assistant"):
                with st.expander(f"📤 tool result · {label}", expanded=False):
                    st.code(str(msg.content))


def _resume(decision: dict[str, Any]) -> None:
    """Send one approval decision back through the interrupt and rerun."""
    result = st.session_state.graph.invoke(
        Command(resume={"decisions": [decision]}), _config()
    )
    st.session_state.result = result
    st.rerun()


def _draft_yaml(draft: dict[str, Any]) -> str:
    try:
        return Contract.model_validate(draft).to_yaml()
    except Exception:
        return yaml.safe_dump(draft, sort_keys=False)


def _render_gate(payload: dict[str, Any]) -> None:
    """The human approval gate: the plain-English contract review plus approve /
    edit / reject controls. Edit exposes the contract YAML for the owner to amend
    before approving — the structured `edit` decision the CLI cannot offer."""
    request = (payload.get("action_requests") or [{}])[0]
    draft = request.get("args", {}).get("contract", {})

    st.divider()
    st.subheader("Approval requested")
    st.markdown(request.get("description", "(no description)"))

    approve_col, reject_col = st.columns(2)
    if approve_col.button("✅ Approve", type="primary", use_container_width=True):
        _resume({"type": "approve"})

    feedback = st.session_state.get("gate_feedback", "")
    if reject_col.button("↩️ Request changes", use_container_width=True):
        _resume({"type": "reject", "message": feedback or "please revise the contract"})
    st.text_input(
        "Changes to request (sent to the agent if you click Request changes):",
        key="gate_feedback",
    )

    with st.expander("✏️ Edit the contract directly, then approve"):
        edited_yaml = st.text_area(
            "Contract YAML", value=_draft_yaml(draft), height=320, key="gate_edit_yaml"
        )
        if st.button("Approve edited contract"):
            try:
                edited = yaml.safe_load(edited_yaml)
            except yaml.YAMLError as exc:
                st.error(f"That YAML did not parse: {exc}")
            else:
                _resume({
                    "type": "edit",
                    "edited_action": {
                        "name": request.get("name", "approve_contract"),
                        "args": {"contract": edited},
                    },
                })


def main() -> None:
    _init_state()
    result = st.session_state.result

    with st.sidebar:
        st.header("dq-agent · scoping")
        show_tools = st.toggle("Show tool calls", value=True)
        st.caption(f"thread `{st.session_state.thread_id}`")

        inp, out = _token_usage(result)
        st.metric("Tokens (in / out)", f"{inp} / {out}")
        st.caption(
            "0 until the Bedrock adapter surfaces `usage_metadata` "
            "(see DeptBedrockChat limitations)."
        )

        if st.button("🔄 New conversation"):
            for key in ("graph", "checkpointer", "thread_id", "result"):
                st.session_state.pop(key, None)
            st.rerun()

    st.title("Scope a data quality contract")
    st.caption(
        "Point me at a dataset (e.g. `data/synthetic/orders.csv`), describe its "
        "business context, and I'll propose a contract for your approval."
    )

    _render_transcript(result, show_tools)

    if (result or {}).get("contract_path"):
        st.success(f"Contract approved and saved to `{result['contract_path']}`")
        return

    payload = _pending_interrupt(result)
    if payload is not None:
        _render_gate(payload)
        return

    prompt = st.chat_input("Message the scoping agent…")
    if prompt:
        st.session_state.result = st.session_state.graph.invoke(
            {"messages": [{"role": "user", "content": prompt}]}, _config()
        )
        st.rerun()


if __name__ == "__main__":
    main()
