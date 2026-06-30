"""Streamlit chat interface for the scoping agent — the non-coder-facing driver.

This is the visual counterpart to `scripts/scoping_cli.py`: same in-process graph,
same checkpointer, same interrupt/resume contract — rendered as a chat panel a
dataset owner can use without a terminal. It mirrors the parts of agent-chat-ui that
matter here, plus a few extras:

  - a chat transcript with the agent;
  - **visible tool calls** with a sidebar **toggle** to show/hide them;
  - **profile-at-a-glance** — the `profile_dataset` result renders as a stats table,
    not raw JSON;
  - the human approval gate as approve / edit / reject controls, with a **rule
    provenance** view (each proposed rule next to the profile signal behind it) and
    **edit** exposing the contract YAML before approval;
  - **resumable threads** — conversations persist in a local SQLite checkpoint store,
    so a closed tab can be reopened and continued;
  - **token-usage**, a dev **session-cost** readout (per-call `x-cost` headers), the
    account **daily budget** (`bedrock.get_usage()`, cached), and a one-click
    **download** of the approved contract YAML.

The graph and its SQLite connection are shared across sessions via `st.cache_resource`
(`_resources`); only per-conversation state lives in `st.session_state`.

Run from the repo root (needs the work environment for the Bedrock proxy to answer):

    uv run --extra ui streamlit run app/scoping_app.py

Not yet wired: token streaming — the Bedrock adapter has no `_stream` (see the
DeptBedrockChat limitations docstring).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from dq_agent.agents.bedrock_chat import BedrockProxyError
from dq_agent.agents.scoping import build_graph
from dq_agent.models import Contract

st.set_page_config(page_title="dq-agent · scoping", page_icon="🧪", layout="centered")

# Local checkpoint store: threads survive a restart so a closed tab can resume.
# Gitignored (*.sqlite). Holds conversation state only — never the scoped data.
DB_PATH = "scoping_threads.sqlite"


@st.cache_resource
def _resources() -> tuple[Any, sqlite3.Connection]:
    """The graph and its checkpoint connection — process-global, built once and shared
    across sessions and reruns. This is exactly the `cache_resource` use case:
    compiling the graph (registry scan, model init) and opening the SQLite connection
    are expensive and safe to share. Per-conversation state (thread_id, transcript)
    stays in session_state, so users remain isolated by thread_id on the shared store.

    check_same_thread=False: a cached connection is reused across Streamlit's worker
    threads. SQLite serializes writes; for many concurrent writers, move to a
    PostgresSaver."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()  # create the checkpoint tables up front so reads don't race
    return build_graph(checkpointer=checkpointer), conn


def _init_state() -> None:
    """Per-session conversation state; the graph/connection are shared via _resources()."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"scoping-{uuid.uuid4().hex[:8]}"
        st.session_state.result = None  # latest graph result; holds the full transcript


def _config() -> dict[str, Any]:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _pending_interrupt(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """The approval-gate payload if the graph is paused, else None."""
    pending = (result or {}).get("__interrupt__")
    if not pending:
        return None
    item = pending[0] if isinstance(pending, (list, tuple)) else pending
    return getattr(item, "value", item)


def _known_threads() -> list[str]:
    """Distinct thread ids in the checkpoint store, for the resume picker. Best-effort:
    the table may not exist yet, and its schema is an internal detail of SqliteSaver."""
    _, conn = _resources()
    try:
        rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        return sorted(r[0] for r in rows)
    except sqlite3.OperationalError:
        return []


def _load_thread(thread_id: str) -> dict[str, Any] | None:
    """Rebuild a result-like dict from a persisted thread so a resumed conversation
    renders (and a pending approval gate re-appears) without re-running the model."""
    graph, _ = _resources()
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        return None
    result = dict(snapshot.values)
    interrupts = [i for task in snapshot.tasks for i in getattr(task, "interrupts", ())]
    if interrupts:
        result["__interrupt__"] = interrupts
    return result


def _token_usage(result: dict[str, Any] | None) -> tuple[int, int]:
    """Sum input/output tokens across the transcript from AIMessage.usage_metadata."""
    inp = out = 0
    for msg in (result or {}).get("messages", []):
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            inp += usage.get("input_tokens", 0)
            out += usage.get("output_tokens", 0)
    return inp, out


def _session_cost(result: dict[str, Any] | None) -> float | None:
    """Total $ across the transcript from response_metadata['cost_usd'], or None when no
    message carries a cost — so the dev metric stays hidden rather than showing a wrong
    $0 when the proxy isn't returning cost yet (see DeptBedrockChat._cost_usd)."""
    total = 0.0
    seen = False
    for msg in (result or {}).get("messages", []):
        meta = getattr(msg, "response_metadata", None) or {}
        if "cost_usd" in meta:
            total += meta["cost_usd"]
            seen = True
    return total if seen else None


@st.cache_data(ttl=60, show_spinner=False)
def _daily_usage() -> dict[str, str] | None:
    """Account-level Bedrock daily budget via `dwutils.bedrock.get_usage()`. Cached for
    a minute (st.cache_data) so it is not re-fetched on every rerun; None off-network."""
    try:
        from dwutils import bedrock

        return bedrock.get_usage()
    except Exception:
        return None


# --- transcript rendering ------------------------------------------------


def _as_profile(content: str) -> dict[str, Any] | None:
    """If a tool result is a profiler report, return it parsed, else None. profile_table
    prefixes a human note before the JSON, so retry from the first brace if the whole
    string does not parse."""
    candidates = [content]
    brace = content.find("{")
    if brace > 0:
        candidates.append(content[brace:])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and "columns" in data and "dataset" in data:
            return data
    return None


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _render_profile(report: dict[str, Any]) -> None:
    """profile-at-a-glance: the redacted report as a stats table, not raw JSON."""
    rows = [
        {
            "column": col["name"],
            "dtype": col["dtype"],
            "null": _pct(col.get("null_rate")),
            "unique": "—" if col.get("uniqueness_ratio") is None
            else f"{col['uniqueness_ratio']:.2f}",
            "hint": col.get("semantic_hint") or "",
        }
        for col in report["columns"]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if report.get("sampled"):
        st.caption(f"sample of {report['table']['row_count']:,} rows — statistics are estimates")
    else:
        st.caption(f"{report['table']['row_count']:,} rows")


def _render_transcript(result: dict[str, Any] | None, show_tools: bool) -> None:
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
            report = _as_profile(str(msg.content))
            with st.chat_message("assistant"):
                if report is not None:
                    with st.expander(f"📊 profile · {report['dataset']}", expanded=True):
                        _render_profile(report)
                else:
                    with st.expander(
                        f"📤 tool result · {msg.name or msg.tool_call_id}", expanded=False
                    ):
                        st.code(str(msg.content))


# --- approval gate -------------------------------------------------------


def _provenance_rows(draft: dict[str, Any], profile: dict[str, Any] | None) -> list[dict]:
    """Each proposed rule next to the profile signal behind it, so the owner sees
    *why* a rule was chosen before approving."""
    cols = {c["name"]: c for c in (profile or {}).get("columns", [])}
    rows = []
    for rule in draft.get("rules", []):
        column = (rule.get("params") or {}).get("column")
        signal = ""
        if column and column in cols:
            col = cols[column]
            bits = []
            if col.get("null_rate"):
                bits.append(f"{col['null_rate'] * 100:.0f}% null")
            if col.get("uniqueness_ratio") is not None:
                bits.append(f"uniq {col['uniqueness_ratio']:.2f}")
            if col.get("semantic_hint"):
                bits.append(f"hint: {col['semantic_hint']}")
            signal = ", ".join(bits)
        rows.append({
            "rule": rule.get("rule_id", "?"),
            "column": column or "(table)",
            "profile signal": signal or "—",
        })
    return rows


def _invoke(graph_input: Any) -> None:
    """Advance the graph and store the result, or render a clear error and stop.

    A bedrock-proxy failure (bad token, model_id, or proxy URL) raises BedrockProxyError;
    catch it so the user sees the diagnostic message in the UI instead of Streamlit's raw
    traceback box, and the existing transcript stays put for a retry."""
    graph, _ = _resources()
    try:
        st.session_state.result = graph.invoke(graph_input, _config())
    except BedrockProxyError as exc:
        st.error(f"The model call failed.\n\n{exc}")
        st.stop()
    st.rerun()


def _resume(decision: dict[str, Any]) -> None:
    _invoke(Command(resume={"decisions": [decision]}))


def _draft_yaml(draft: dict[str, Any]) -> str:
    try:
        return Contract.model_validate(draft).to_yaml()
    except Exception:
        return yaml.safe_dump(draft, sort_keys=False)


def _render_gate(payload: dict[str, Any], profile: dict[str, Any] | None) -> None:
    request = (payload.get("action_requests") or [{}])[0]
    draft = request.get("args", {}).get("contract", {})

    st.divider()
    st.subheader("Approval requested")
    st.markdown(request.get("description", "(no description)"))

    with st.expander("🔎 Why these rules? (profile signals)"):
        st.dataframe(
            _provenance_rows(draft, profile), use_container_width=True, hide_index=True
        )

    approve_col, reject_col = st.columns(2)
    if approve_col.button("✅ Approve", type="primary", use_container_width=True):
        _resume({"type": "approve"})
    if reject_col.button("↩️ Request changes", use_container_width=True):
        feedback = st.session_state.get("gate_feedback") or "please revise the contract"
        _resume({"type": "reject", "message": feedback})
    st.text_input(
        "Changes to request (sent if you click Request changes):", key="gate_feedback"
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


# --- page ----------------------------------------------------------------


def _sidebar(result: dict[str, Any] | None) -> bool:
    with st.sidebar:
        st.header("dq-agent · scoping")
        show_tools = st.toggle("Show tool calls", value=True)
        st.caption(f"thread `{st.session_state.thread_id}`")

        inp, out = _token_usage(result)
        st.metric("Tokens (in / out)", f"{inp} / {out}")
        st.caption("Summed from the Bedrock response usage.")

        cost = _session_cost(result)
        if cost is not None:
            st.metric("Session cost (dev)", f"${cost:.4f}")

        usage = _daily_usage()
        if usage:
            st.divider()
            st.caption("Bedrock daily budget")
            st.write(f"{usage['Current Daily Used']} of {usage['Daily Limit']} used")
            st.caption(f"{usage['Daily Usage Remaining']} remaining")

        st.divider()
        threads = [t for t in _known_threads() if t != st.session_state.thread_id]
        if threads:
            choice = st.selectbox("Resume a saved thread", threads, index=None)
            if choice and st.button("Open thread"):
                st.session_state.thread_id = choice
                st.session_state.result = _load_thread(choice)
                st.rerun()
        if st.button("🔄 New conversation"):
            st.session_state.thread_id = f"scoping-{uuid.uuid4().hex[:8]}"
            st.session_state.result = None
            st.rerun()
    return show_tools


def main() -> None:
    _init_state()
    result = st.session_state.result
    show_tools = _sidebar(result)

    st.title("Scope a data quality contract")
    st.caption(
        "Point me at a dataset — a CSV/Parquet path (e.g. `data/synthetic/orders.csv`) "
        "or a Postgres table (e.g. `public.orders`) — describe its business context, "
        "and I'll propose a contract for your approval."
    )

    _render_transcript(result, show_tools)

    contract_path = (result or {}).get("contract_path")
    if contract_path:
        st.success(f"Contract approved and saved to `{contract_path}`")
        try:
            yaml_text = Path(contract_path).read_text()
            st.download_button(
                "⬇️ Download contract YAML",
                yaml_text,
                file_name=Path(contract_path).name,
                mime="application/x-yaml",
            )
        except OSError:
            pass
        return

    payload = _pending_interrupt(result)
    if payload is not None:
        _render_gate(payload, (result or {}).get("profile"))
        return

    prompt = st.chat_input("Message the scoping agent…")
    if prompt:
        _invoke({"messages": [{"role": "user", "content": prompt}]})


if __name__ == "__main__":
    main()
