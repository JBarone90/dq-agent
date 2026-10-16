"""The terminal driver, `scripts/scoping_cli.py`, against a scripted model."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("great_expectations")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from dq_agent.agents.bedrock_chat import BedrockProxyError
from dq_agent.agents.scoping import build_graph

_SPEC = importlib.util.spec_from_file_location(
    "scoping_cli", Path(__file__).parent.parent / "scripts" / "scoping_cli.py")
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _call(name, args, call_id):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def _to_gate(synthetic_data_path):
    """The model's side of a session that reaches the approval gate."""
    return [
        AIMessage("Which file?"),
        _call("profile_dataset", {"path": str(synthetic_data_path / "orders.csv")}, "c1"),
        _call("propose_contract", {"table": "synthetic.orders", "expectations": [
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "order_id"}}]}, "c2"),
        _call("request_approval", {}, "c3"),
    ]


def _graph(tmp_path, registry, script):
    return build_graph(model=ScriptedModel(messages=iter(script)), registry=registry,
                       contracts_dir=tmp_path, proposals_dir=tmp_path, checkpointer=InMemorySaver())


def _typing(monkeypatch, *lines):
    answers = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)


def test_a_session_ends_with_an_approved_contract(tmp_path, registry, synthetic_data_path, monkeypatch, capsys):
    monkeypatch.setenv("DQ_AGENT_APPROVER", "owner")
    graph = _graph(tmp_path, registry, [*_to_gate(synthetic_data_path), AIMessage("saved")])
    _typing(monkeypatch, "hello", "orders.csv please", "approve")

    cli.converse(graph, "t1")

    out = capsys.readouterr().out
    assert "Agent: Which file?" in out
    assert "APPROVAL REQUESTED" in out and "`order_id` must never be empty" in out
    assert "Contract approved and saved to" in out
    assert (tmp_path / "synthetic.orders.yaml").exists()


def test_feedback_at_the_gate_goes_back_to_the_agent(tmp_path, registry, synthetic_data_path, monkeypatch, capsys):
    graph = _graph(tmp_path, registry, [*_to_gate(synthetic_data_path), AIMessage("Revising")])
    _typing(monkeypatch, "hello", "orders.csv", "make it a warning")

    with pytest.raises(KeyboardInterrupt):  # input runs out after the agent's reply
        cli.converse(graph, "t1")

    assert "Agent: Revising" in capsys.readouterr().out
    last_tool = [m for m in graph.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
                 if m.type == "tool"][-1]
    assert "feedback: make it a warning" in last_tool.content
    assert not (tmp_path / "synthetic.orders.yaml").exists()


def test_a_reopened_thread_resumes_at_its_gate(tmp_path, registry, synthetic_data_path, monkeypatch):
    """Quitting at the gate and coming back must answer the gate, not send a new message."""
    graph = _graph(tmp_path, registry, [*_to_gate(synthetic_data_path), AIMessage("saved")])
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"messages": [HumanMessage("hello")]}, config)
    graph.invoke({"messages": [HumanMessage("orders.csv")]}, config)  # now paused at the gate

    _typing(monkeypatch, "approve")
    cli.converse(graph, "t1")

    humans = [m for m in graph.get_state(config).values["messages"] if isinstance(m, HumanMessage)]
    assert [m.content for m in humans] == ["hello", "orders.csv"]
    assert (tmp_path / "synthetic.orders.yaml").exists()


@pytest.mark.parametrize("typed", ["quit", "EXIT", ":q"])
def test_quit_words_stop(monkeypatch, typed):
    _typing(monkeypatch, typed)
    with pytest.raises(KeyboardInterrupt):
        cli._prompt("You:")


def test_end_of_input_stops(monkeypatch):
    _typing(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        cli._prompt("You:")


def test_last_ai_text_skips_tool_calls_and_reads_block_content():
    messages = [AIMessage("first"), _call("list_rules", {}, "c1")]
    assert cli._last_ai_text({"messages": messages}) == "first"
    assert cli._last_ai_text({"messages": [AIMessage([{"type": "text", "text": "x"}])]}).startswith("[")
    assert cli._last_ai_text({"messages": []}) == ""


def test_sqlite_checkpointer_keeps_a_thread_across_runs(tmp_path, registry, synthetic_data_path):
    db = str(tmp_path / "threads.sqlite")
    config = {"configurable": {"thread_id": "t1"}}
    with cli._checkpointer(db) as saver:
        graph = build_graph(model=ScriptedModel(messages=iter([AIMessage("hi")])), registry=registry,
                            contracts_dir=tmp_path, checkpointer=saver)
        graph.invoke({"messages": [HumanMessage("hello")]}, config)
    with cli._checkpointer(db) as saver:
        graph = build_graph(model=ScriptedModel(messages=iter([])), registry=registry,
                            contracts_dir=tmp_path, checkpointer=saver)
        assert [m.content for m in graph.get_state(config).values["messages"]] == ["hello", "hi"]


def test_memory_checkpointer_by_default():
    with cli._checkpointer(None) as saver:
        assert isinstance(saver, InMemorySaver)


class _FailingGraph:
    def invoke(self, *args, **kwargs):
        raise BedrockProxyError("bedrock-proxy call failed: 403", 403)

    def get_state(self, config):
        class Snapshot:
            values = {}
            tasks = ()
        return Snapshot()


def test_main_reports_a_proxy_failure_with_how_to_resume(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "build_graph", lambda **kwargs: _FailingGraph())
    monkeypatch.setattr("sys.argv", ["scoping_cli.py", "--db", str(tmp_path / "t.sqlite"), "--thread", "abc"])
    _typing(monkeypatch, "hello")

    cli.main()

    out = capsys.readouterr().out
    assert "The model call failed" in out and "403" in out
    assert "--thread abc" in out


def test_main_stops_cleanly_on_quit(monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_graph", lambda **kwargs: _FailingGraph())
    monkeypatch.setattr("sys.argv", ["scoping_cli.py"])
    _typing(monkeypatch, "quit")

    cli.main()

    assert "Pass --db to make a session resumable" in capsys.readouterr().out
