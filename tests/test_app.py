"""The Streamlit driver, `app/scoping_app.py`, run headless with Streamlit's AppTest.

The app builds its graph through `build_graph`, patched here to a scripted model so no proxy is
needed. It runs from a temporary directory: its thread store and data directory are relative.
"""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("langgraph")
pytest.importorskip("great_expectations")
pytest.importorskip("langchain_anthropic")

import anthropic
import httpx2
import streamlit as st
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from streamlit.testing.v1 import AppTest

from dq_agent.agents import scoping

APP = str(Path(__file__).parent.parent / "app" / "scoping_app.py")
REPO = Path(__file__).parent.parent


def _forbidden():
    """What the SDK raises when the proxy answers 403 (e.g. the daily budget is spent)."""
    request = httpx2.Request("POST", "http://proxy/anthropic/v1/messages")
    return anthropic.PermissionDeniedError(
        "Error code: 403 - daily budget spent", response=httpx2.Response(403, request=request), body=None)


class ScriptedModel(GenericFakeChatModel):
    """Plays back AI messages; streams each as a single chunk (the app streams by default, and the
    stock fake cannot stream a message that is only tool calls)."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = next(self.messages)
        yield ChatGenerationChunk(message=AIMessageChunk(
            content=message.content,
            tool_call_chunks=[{"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i,
                               "type": "tool_call_chunk"} for i, c in enumerate(message.tool_calls)],
        ))


def _call(name, args, call_id):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


TO_GATE = [
    _call("profile_dataset", {"path": "data/orders.csv"}, "c1"),
    _call("propose_contract", {"table": "synthetic.orders", "expectations": [
        {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "order_id"},
         "meta": {"severity": "warning"}}]}, "c2"),
    _call("request_approval", {}, "c3"),
]


@pytest.fixture
def workdir(tmp_path, monkeypatch, registry):
    (tmp_path / "data").mkdir()
    shutil.copy(REPO / "data" / "synthetic" / "orders.csv", tmp_path / "data" / "orders.csv")
    monkeypatch.chdir(tmp_path)
    st.cache_resource.clear()
    yield tmp_path
    st.cache_resource.clear()


def _app(monkeypatch, registry, script, workdir):
    real_build = scoping.build_graph

    def build(**kwargs):
        return real_build(model=ScriptedModel(messages=iter(script)), registry=registry,
                          contracts_dir=workdir / "contracts", proposals_dir=workdir / "proposals", **kwargs)

    monkeypatch.setattr(scoping, "build_graph", build)
    return AppTest.from_file(APP, default_timeout=60).run()


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _scope(at, path="data/orders.csv"):
    at.text_input[0].input(path).run()
    _button(at, "Scope this").click().run()
    return at


def test_first_page_asks_for_a_dataset(monkeypatch, registry, workdir):
    at = _app(monkeypatch, registry, [], workdir)
    assert not at.exception
    assert at.subheader[0].value == "Start by choosing a dataset"
    assert _button(at, "Scope this").disabled


@pytest.mark.parametrize("path, message", [
    ("data/orders.txt", "Path must end in .csv or .parquet."),
    ("../orders.csv", "The file must be under `data/`."),
    ("data/missing.csv", "No file at that path"),
])
def test_a_bad_path_is_caught_before_any_model_call(monkeypatch, registry, workdir, path, message):
    at = _app(monkeypatch, registry, [], workdir)
    at.text_input[0].input(path).run()
    assert any(message in c.value for c in at.caption)
    assert _button(at, "Scope this").disabled


def test_a_table_name_must_be_schema_qualified_form(monkeypatch, registry, workdir):
    monkeypatch.delenv("DATABASE_DSN__datasets_1", raising=False)
    at = _app(monkeypatch, registry, [], workdir)
    at.radio[0].set_value("Postgres table").run()
    assert any("DATABASE_DSN__datasets_1" in i.value for i in at.info)
    at.text_input[0].input("bad name!").run()
    assert any("Use the form schema.table" in c.value for c in at.caption)
    at.text_input[0].input("public.orders").run()
    assert not _button(at, "Scope this").disabled


def test_scoping_reaches_the_gate_and_approval_saves_the_contract(monkeypatch, registry, workdir):
    monkeypatch.setenv("DQ_AGENT_APPROVER", "owner")
    at = _scope(_app(monkeypatch, registry, [*TO_GATE, AIMessage("Saved.")], workdir))
    assert not at.exception
    assert any(s.value == "Approval requested" for s in at.subheader)
    assert any("**warning**" in m.value for m in at.markdown)

    _button(at, "✅ Approve").click().run()

    assert not at.exception
    assert any("Contract approved and saved" in s.value for s in at.success)
    assert (workdir / "contracts" / "synthetic.orders.yaml").exists()


def test_requesting_changes_sends_the_feedback(monkeypatch, registry, workdir):
    at = _scope(_app(monkeypatch, registry, [*TO_GATE, AIMessage("I will revise it.")], workdir))
    _button(at, "↩️ Request changes").click().run()
    next(t for t in at.text_area if t.label == "What should change?").input("loosen it").run()
    _button(at, "Send").click().run()

    assert not at.exception
    assert any("I will revise it." in m.value for m in at.markdown)
    assert not (workdir / "contracts").exists()


def test_editing_the_yaml_approves_the_edited_contract(monkeypatch, registry, workdir):
    monkeypatch.setenv("DQ_AGENT_APPROVER", "owner")
    at = _scope(_app(monkeypatch, registry, [*TO_GATE, AIMessage("Saved.")], workdir))
    editor = next(t for t in at.text_area if t.label == "Contract YAML")
    editor.input(editor.value.replace("severity: warning", "severity: error")).run()
    _button(at, "Approve edited contract").click().run()

    assert not at.exception
    assert "severity: error" in (workdir / "contracts" / "synthetic.orders.yaml").read_text(encoding="utf-8")


def test_unparsable_yaml_is_reported_not_sent(monkeypatch, registry, workdir):
    at = _scope(_app(monkeypatch, registry, [*TO_GATE], workdir))
    next(t for t in at.text_area if t.label == "Contract YAML").input("suite: [unclosed").run()
    _button(at, "Approve edited contract").click().run()
    assert any("That YAML did not parse" in e.value for e in at.error)


def test_without_streaming_the_gate_still_appears(monkeypatch, registry, workdir):
    at = _app(monkeypatch, registry, [*TO_GATE], workdir)
    next(t for t in at.toggle if t.label == "Stream responses").set_value(False).run()
    _scope(at)
    assert any(s.value == "Approval requested" for s in at.subheader)


def test_plain_code_mode_renders_the_contract_as_text(monkeypatch, registry, workdir):
    at = _app(monkeypatch, registry, [*TO_GATE], workdir)
    next(t for t in at.toggle if t.label == "Plain-text code blocks").set_value(True).run()
    _scope(at)
    assert any("expect_column_values_to_not_be_null" in t.value for t in at.text)


def test_a_proxy_failure_is_shown_not_raised(monkeypatch, registry, workdir):
    class Failing(ScriptedModel):
        def _generate(self, *args, **kwargs):
            raise _forbidden()

        def _stream(self, *args, **kwargs):
            raise _forbidden()

    real_build = scoping.build_graph
    monkeypatch.setattr(scoping, "build_graph", lambda **kwargs: real_build(
        model=Failing(messages=iter([])), registry=registry, contracts_dir=workdir, **kwargs))
    at = _scope(AppTest.from_file(APP, default_timeout=60).run())
    assert any("The model call failed" in e.value for e in at.error)


def test_a_saved_thread_can_be_reopened(monkeypatch, registry, workdir):
    at = _scope(_app(monkeypatch, registry, [*TO_GATE], workdir))
    first = at.session_state.thread_id
    _button(at, "🔄 New conversation").click().run()
    assert at.session_state.thread_id != first
    picker = next(s for s in at.selectbox if s.label == "Resume a saved thread")
    picker.set_value(first).run()
    _button(at, "Open thread").click().run()
    assert at.session_state.thread_id == first
    assert any(s.value == "Approval requested" for s in at.subheader)  # the gate comes back with it


def test_the_profile_renders_as_a_table(monkeypatch, registry, workdir):
    at = _scope(_app(monkeypatch, registry, [*TO_GATE], workdir))
    assert at.dataframe, "the profile tool result should render as a stats table"
    assert any("20 rows" in c.value for c in at.caption)
