"""Streamlit chat interface for the scoping agent: the non-coder-facing driver.

The visual counterpart to `scripts/scoping_cli.py`: same in-process graph, checkpointer
and interrupt/resume contract, rendered as a chat panel. It shows the transcript with an
optional view of tool calls, the profile as a stats table, the approval gate as
approve / edit / reject controls, resumable threads (a local SQLite checkpoint store),
token usage, and a download of the approved contract YAML.

The graph and its SQLite connection are shared across sessions via `st.cache_resource`
(`_resources`); only per-conversation state lives in `st.session_state`.

Run from the repo root (needs the work environment for the bedrock-proxy to answer):

    uv run --extra ui streamlit run app/scoping_app.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver

from dq_agent.agents.gate import pending_interrupt, resume
from dq_agent.agents.llm import ModelError
from dq_agent.agents.scoping import DEFAULT_DATA_DIR, build_graph
from dq_agent.connectors import DEFAULT_DSN_ENV, is_valid_table
from dq_agent.models import Contract

_FILE_SUFFIXES = {".csv", ".parquet"}

st.set_page_config(page_title="dq-agent · scoping", page_icon="🧪", layout="centered")

# Local checkpoint store: threads survive a restart so a closed tab can resume.
# Gitignored (*.sqlite). Holds conversation state only — never the scoped data.
DB_PATH = "scoping_threads.sqlite"


@st.cache_resource
def _resources() -> tuple[Any, sqlite3.Connection]:
    """The compiled graph and its SQLite connection, shared across sessions and reruns.
    Per-conversation state stays in session_state, keyed by thread_id on the shared store.
    check_same_thread=False because Streamlit reuses the connection across worker threads;
    move to a PostgresSaver for many concurrent writers."""
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


# --- transcript rendering ------------------------------------------------


def _as_profile(content: str) -> dict[str, Any] | None:
    """If a tool result is a profiler report, return it parsed, else None. Both profile
    tools emit the report as pure JSON, so a single parse suffices."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
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
        est = report["table"].get("estimated_total_rows")
        of = f" of an estimated {est:,}" if est else ""
        st.caption(
            f"sample of {report['table']['row_count']:,} rows{of} — statistics are estimates"
        )
    else:
        st.caption(f"{report['table']['row_count']:,} rows")


_FENCE = re.compile(r"```[A-Za-z]*\n(.*?)```", re.DOTALL)


def _plain_code() -> bool:
    """True when code blocks should render without syntax highlighting.

    Streamlit lazy-loads a 1.5 MB `StreamlitSyntaxHighlighter` chunk the first time a
    page renders `st.code`, `st.json`, or a fenced block inside `st.markdown`. Some
    locked-down networks never deliver a chunk that size, and the component then dies
    with "Failed to fetch dynamically imported module" - which lands squarely on the
    approval gate, where the contract YAML has to be readable for the gate to mean
    anything. This drops to `st.text`, which needs no chunk at all."""
    return bool(st.session_state.get("plain_code"))


def _code(text: str, language: str | None = None) -> None:
    """A code block that degrades to plain text when the highlighter cannot load."""
    if _plain_code():
        st.text(text)
    else:
        st.code(text, language=language)


def _markdown(text: str) -> None:
    """`st.markdown`, but in plain mode fenced blocks are peeled out and rendered as
    text, so the surrounding prose still renders as markdown and the highlighter chunk
    is never requested."""
    if not _plain_code():
        st.markdown(text)
        return
    pos = 0
    for match in _FENCE.finditer(text):
        before = text[pos:match.start()]
        if before.strip():
            st.markdown(before)
        st.text(match.group(1).rstrip())
        pos = match.end()
    tail = text[pos:]
    if tail.strip():
        st.markdown(tail)


def _render_transcript(result: dict[str, Any] | None, show_tools: bool) -> None:
    for msg in (result or {}).get("messages", []):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            # `.text`: the content is a list of blocks, thinking and tool use included
            if msg.text:
                with st.chat_message("assistant"):
                    st.markdown(msg.text)
            if show_tools and msg.tool_calls:
                with st.chat_message("assistant"):
                    for call in msg.tool_calls:
                        with st.expander(f"🔧 tool call · {call['name']}", expanded=False):
                            _code(json.dumps(call["args"], indent=2), "json")
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
                        _code(str(msg.content))


# --- approval gate -------------------------------------------------------


def _agent_tokens(graph: Any, graph_input: Any) -> Any:
    """Yield assistant prose as the graph produces it, for `st.write_stream`. Tool-call
    chunks (partial JSON) are skipped; the transcript is re-rendered from state afterwards."""
    for chunk, _meta in graph.stream(graph_input, _config(), stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk) and chunk.text:
            yield chunk.text


def _invoke(graph_input: Any) -> None:
    """Advance the graph and store the result, or show a ModelError (also raised
    mid-stream) in the UI instead of a traceback, leaving the transcript for a retry.
    After streaming, state is reloaded from the checkpointer: a pending approval gate
    only appears in the persisted snapshot."""
    graph, _ = _resources()
    try:
        if st.session_state.get("stream_responses", True):
            with st.chat_message("assistant"):
                st.write_stream(_agent_tokens(graph, graph_input))
            st.session_state.result = (
                _load_thread(st.session_state.thread_id) or st.session_state.result
            )
        else:
            st.session_state.result = graph.invoke(graph_input, _config())
    except ModelError as exc:
        st.error(f"The model call failed.\n\n{exc}")
        st.stop()
    st.rerun()


def _resume(decision: dict[str, Any]) -> None:
    _invoke(resume(decision))


def _draft_yaml(draft: dict[str, Any]) -> str:
    try:
        return Contract.model_validate(draft).to_yaml()
    except ValueError:  # pydantic's ValidationError: show the draft as it is
        return yaml.safe_dump(draft, sort_keys=False)


def _render_gate(payload: dict[str, Any]) -> None:
    request = (payload.get("action_requests") or [{}])[0]
    draft = request.get("args", {}).get("contract", {})
    # Keyed to this draft: a keyed widget keeps what it holds across reruns even when its
    # value= changes, so a fixed key would leave a revised draft's editor showing the old
    # YAML, and approving the edit would approve the old contract.
    draft_key = hashlib.sha256(json.dumps(draft, sort_keys=True, default=str).encode()).hexdigest()[:12]

    st.divider()
    st.subheader("Approval requested")
    _markdown(request.get("description", "(no description)"))

    approve_col, reject_col = st.columns(2)
    if approve_col.button("✅ Approve", type="primary", use_container_width=True):
        _resume({"type": "approve"})
    if reject_col.button("↩️ Request changes", use_container_width=True):
        st.session_state.gate_requesting_changes = True

    if st.session_state.get("gate_requesting_changes"):
        feedback = st.text_area("What should change?", key=f"gate_feedback_{draft_key}")
        send_col, cancel_col = st.columns(2)
        if send_col.button("Send", type="primary", use_container_width=True):
            st.session_state.gate_requesting_changes = False
            _resume({"type": "reject", "message": feedback or "please revise the contract"})
        if cancel_col.button("Cancel", use_container_width=True):
            st.session_state.gate_requesting_changes = False
            st.rerun()

    with st.expander("✏️ Edit the contract directly, then approve"):
        edited_yaml = st.text_area(
            "Contract YAML", value=_draft_yaml(draft), height=320, key=f"gate_edit_yaml_{draft_key}"
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
        st.toggle(
            "Stream responses",
            value=True,
            key="stream_responses",
            help="Render the reply as it is produced.",
        )
        st.toggle(
            "Plain-text code blocks",
            key="plain_code",
            help=(
                "Turn on if code blocks fail with 'Failed to fetch dynamically "
                "imported module'. Streamlit's syntax highlighter is a 1.5 MB lazy "
                "chunk that some restricted networks will not deliver."
            ),
        )
        st.caption(f"thread `{st.session_state.thread_id}`")

        inp, out = _token_usage(result)
        st.metric("Tokens (in / out)", f"{inp} / {out}")
        st.caption("Summed from the model's reported usage.")


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


def _validate_path(path: str) -> tuple[bool, str]:
    """Client-side check for a file locator: a hint, not the authority (load_csv /
    load_parquet still validate). Empty input is 'not yet valid' with no scolding."""
    if not path:
        return False, ""
    p = Path(path)
    if p.suffix.lower() not in _FILE_SUFFIXES:
        return False, "Path must end in .csv or .parquet."
    if not p.resolve().is_relative_to(DEFAULT_DATA_DIR.resolve()):
        return False, f"The file must be under `{DEFAULT_DATA_DIR}/`."
    if not p.exists():
        return False, "No file at that path (relative to where the app runs)."
    return True, ""


def _source_entry() -> None:
    """First-turn onboarding: help the owner enter a valid locator before any model
    call. Shown only before a conversation exists; the chat box below stays an escape
    hatch. Validation is client-side so a typo is caught without a model round-trip —
    the profile tools validate authoritatively when they run."""
    st.subheader("Start by choosing a dataset")
    kind = st.radio(
        "Source", ["File (CSV / Parquet)", "Postgres table"],
        horizontal=True, label_visibility="collapsed",
    )
    if kind.startswith("File"):
        path = st.text_input(
            "File path", placeholder="data/synthetic/orders.csv",
            help=f"A CSV or Parquet file under `{DEFAULT_DATA_DIR}/`.",
        ).strip()
        valid, message = _validate_path(path)
        first_message = f"Please scope the dataset at {path}"
    else:
        if not os.environ.get(DEFAULT_DSN_ENV):
            st.info(f"Set `{DEFAULT_DSN_ENV}` in the environment to scope a database table.")
        table = st.text_input(
            "Table", placeholder="public.orders",
            help="Schema-qualified name, e.g. public.orders.",
        ).strip()
        valid = is_valid_table(table)
        message = "" if valid or not table else "Use the form schema.table, e.g. public.orders."
        first_message = f"Please scope the Postgres table {table}"

    if message:
        st.caption(f"⚠️ {message}")
    if st.button("Scope this", type="primary", disabled=not valid):
        _invoke({"messages": [{"role": "user", "content": first_message}]})


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
            yaml_text = Path(contract_path).read_text(encoding="utf-8")
            st.download_button(
                "⬇️ Download contract YAML",
                yaml_text,
                file_name=Path(contract_path).name,
                mime="application/x-yaml",
            )
        except OSError:
            pass
        return

    payload = pending_interrupt(result)
    if payload is not None:
        _render_gate(payload)
        return

    if result is None:
        # guided first-turn locator only — chat opens up once a conversation exists,
        # so the owner can't type a stray message before choosing a source
        _source_entry()
        return

    prompt = st.chat_input("Message the scoping agent…")
    if prompt:
        _invoke({"messages": [{"role": "user", "content": prompt}]})


if __name__ == "__main__":
    main()