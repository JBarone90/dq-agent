"""Scoping agent tests: tools are exercised directly; the graph (routing, approval
interrupt, contract persistence) is exercised with a scripted fake model — no LLM
calls anywhere in the suite."""

import json

import pytest

pytest.importorskip("langgraph")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import dq_agent.connectors as connectors
from dq_agent.agents.scoping import ProposedRule, _make_tools, build_graph
from dq_agent.connectors import ProfilingLoad
from dq_agent.engine import run
from dq_agent.models import Contract


class ScriptedModel(GenericFakeChatModel):
    """Plays back a fixed sequence of AI messages; accepts any tool binding."""

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def tools(registry):
    return {t.name: t for t in _make_tools(registry)}


def _tool_call(name, args, call_id="c1"):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


# --- tools ---------------------------------------------------------------


def test_profile_dataset_returns_redacted_profile(tools, synthetic_data_path):
    command = tools["profile_dataset"].invoke(
        _tool_call("profile_dataset", {"path": str(synthetic_data_path / "orders.csv")})
    )
    profile = command.update["profile"]
    assert profile["dataset"] == "orders"
    assert profile["redacted"] is True
    assert all(c["top_values"] is None for c in profile["columns"])
    # known raw cell values from the synthetic data must not reach the LLM
    message = command.update["messages"][0]
    assert "not-an-email" not in message.content


def test_profile_dataset_missing_file_errors(tools):
    command = tools["profile_dataset"].invoke(
        _tool_call("profile_dataset", {"path": "nope/missing.csv"})
    )
    assert "error" in command.update["messages"][0].content
    assert "profile" not in command.update


def test_list_rules_filters_by_tag(tools):
    catalogue = tools["list_rules"].invoke({"tags": ["completeness"]})
    assert "null_check" in catalogue
    assert "unique_check" not in catalogue


def test_list_rules_unfiltered_lists_registry(tools, registry):
    catalogue = tools["list_rules"].invoke({})
    for rule_id in registry.rule_ids:
        assert rule_id in catalogue


def test_propose_contract_requires_profile(tools):
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"rules": [{"rule_id": "null_check", "params": {"column": "order_id"}}],
         "state": {}},
    ))
    assert "profile the dataset first" in command.update["messages"][0].content


def test_propose_contract_rejects_unknown_rule(tools, synthetic_data_path):
    profiled = tools["profile_dataset"].invoke(
        _tool_call("profile_dataset", {"path": str(synthetic_data_path / "orders.csv")})
    )
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"rules": [{"rule_id": "does_not_exist", "params": {}}],
         "state": {"profile": profiled.update["profile"]}},
    ))
    assert "unknown rule 'does_not_exist'" in command.update["messages"][0].content
    assert "draft" not in command.update


def test_propose_contract_records_unapproved_draft(tools, synthetic_data_path, orders_df):
    profiled = tools["profile_dataset"].invoke(
        _tool_call("profile_dataset", {"path": str(synthetic_data_path / "orders.csv")})
    )
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"rules": [{"rule_id": "null_check", "params": {"column": "order_id"},
                    "severity": "warning"}],
         "state": {"profile": profiled.update["profile"]}},
    ))
    draft = Contract.model_validate(command.update["draft"])
    assert draft.approved_at is None
    assert draft.dataset == "orders"
    assert draft.rules[0].severity == "warning"
    # schema snapshot comes from the profile, so the engine's drift check will hold
    assert draft.columns == {name: str(dtype) for name, dtype in orders_df.schema.items()}


# --- postgres profiling + gated bounds confirmation ----------------------


def _mock_pg(monkeypatch, df, *, sampled, estimated_rows, bounds=(0, 0), bounds_spy=None):
    """Stub the Postgres path: no live DB. `bounds_spy` records every column
    column_bounds is asked about, so a test can assert it ran only when it should."""
    monkeypatch.setattr(connectors, "resolve_dsn", lambda *a, **k: "postgresql://x/y")
    monkeypatch.setattr(
        connectors, "load_postgres_profiling",
        lambda *a, **k: ProfilingLoad(df, sampled, estimated_rows),
    )

    def fake_bounds(uri, table, column):
        if bounds_spy is not None:
            bounds_spy.append(column)
        return bounds

    monkeypatch.setattr(connectors, "column_bounds", fake_bounds)


def test_profile_table_flags_sampled_report(tools, orders_df, monkeypatch):
    _mock_pg(monkeypatch, orders_df, sampled=True, estimated_rows=5_000_000)
    command = tools["profile_table"].invoke(
        _tool_call("profile_table", {"table": "public.orders"})
    )
    profile = command.update["profile"]
    assert profile["sampled"] is True
    # surplus does not extrapolate from a sample, so the profiler drops it
    assert profile["table"]["duplicate_row_count"] is None
    # the full-table estimate rides in the report itself, not a prose note
    assert profile["table"]["estimated_total_rows"] == 5_000_000
    assert command.update["source_table"] == "public.orders"
    # same output shape as profile_dataset: the message is the report as pure JSON
    assert json.loads(command.update["messages"][0].content)["sampled"] is True


def test_profile_table_full_load_not_sampled(tools, orders_df, monkeypatch):
    _mock_pg(monkeypatch, orders_df, sampled=False, estimated_rows=20)
    command = tools["profile_table"].invoke(
        _tool_call("profile_table", {"table": "public.orders"})
    )
    assert command.update["profile"]["sampled"] is False
    # a full load is authoritative: no population estimate carried
    assert command.update["profile"]["table"]["estimated_total_rows"] is None
    assert json.loads(command.update["messages"][0].content)["sampled"] is False


def test_profile_table_surfaces_connector_error(tools, monkeypatch):
    def boom(*a, **k):
        raise KeyError("DATABASE_DSN__datasets_1")

    monkeypatch.setattr(connectors, "resolve_dsn", boom)
    command = tools["profile_table"].invoke(
        _tool_call("profile_table", {"table": "public.orders"})
    )
    assert "error" in command.update["messages"][0].content
    assert "profile" not in command.update  # nothing recorded on failure


def test_propose_contract_confirms_bounds_when_sampled(tools, orders_df, monkeypatch):
    spy = []
    _mock_pg(monkeypatch, orders_df, sampled=True, estimated_rows=5_000_000,
             bounds=(-200, 99999), bounds_spy=spy)
    profiled = tools["profile_table"].invoke(
        _tool_call("profile_table", {"table": "public.orders"})
    )
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"rules": [{"rule_id": "range_check",
                    "params": {"column": "amount", "min_val": 0, "max_val": 5000}}],
         "state": {"profile": profiled.update["profile"], "source_table": "public.orders"}},
    ))
    assert spy == ["amount"]  # confirmed exactly the referenced column, once
    content = command.update["messages"][0].content
    assert "existing rows would violate" in content
    assert "99999" in content  # the real max, above the proposed max_val


def test_propose_contract_skips_bounds_when_not_sampled(tools, orders_df, monkeypatch):
    spy = []
    _mock_pg(monkeypatch, orders_df, sampled=False, estimated_rows=20,
             bounds=(-200, 99999), bounds_spy=spy)
    profiled = tools["profile_table"].invoke(
        _tool_call("profile_table", {"table": "public.orders"})
    )
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"rules": [{"rule_id": "range_check",
                    "params": {"column": "amount", "min_val": 0, "max_val": 5000}}],
         "state": {"profile": profiled.update["profile"], "source_table": "public.orders"}},
    ))
    assert spy == []  # full-load profile: no confirmation query
    assert "existing rows would violate" not in command.update["messages"][0].content


def test_propose_contract_skips_bounds_without_range_check(tools, orders_df, monkeypatch):
    spy = []
    _mock_pg(monkeypatch, orders_df, sampled=True, estimated_rows=5_000_000,
             bounds=(-200, 99999), bounds_spy=spy)
    profiled = tools["profile_table"].invoke(
        _tool_call("profile_table", {"table": "public.orders"})
    )
    tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"rules": [{"rule_id": "null_check", "params": {"column": "order_id"}}],
         "state": {"profile": profiled.update["profile"], "source_table": "public.orders"}},
    ))
    assert spy == []  # no range_check -> nothing to confirm


# --- proposed rule param coercion ----------------------------------------


def test_proposed_rule_coerces_json_string_params():
    """Some models serialize params as a JSON string; the validator parses it so the
    first tool call validates instead of needing a retry."""
    rule = ProposedRule(rule_id="null_check", params='{"column": "email"}')
    assert rule.params == {"column": "email"}


def test_proposed_rule_keeps_dict_params_untouched():
    rule = ProposedRule(rule_id="null_check", params={"column": "email"})
    assert rule.params == {"column": "email"}


def test_propose_contract_accepts_string_params(tools, synthetic_data_path):
    """End to end through the tool: string params no longer fail validation."""
    profiled = tools["profile_dataset"].invoke(
        _tool_call("profile_dataset", {"path": str(synthetic_data_path / "orders.csv")})
    )
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"rules": [{"rule_id": "unique_check", "params": '{"column": "order_id"}'}],
         "state": {"profile": profiled.update["profile"]}},
    ))
    draft = Contract.model_validate(command.update["draft"])
    assert draft.rules[0].params == {"column": "order_id"}


# --- graph: approval gate ------------------------------------------------


def _scoping_graph(tmp_path, registry, synthetic_data_path, closing_message):
    csv = str(synthetic_data_path / "orders.csv")
    script = [
        AIMessage("", tool_calls=[_tool_call("profile_dataset", {"path": csv}, "c1")]),
        AIMessage("", tool_calls=[_tool_call(
            "propose_contract",
            {"rules": [{"rule_id": "null_check", "params": {"column": "order_id"}}]},
            "c2",
        )]),
        AIMessage("", tool_calls=[_tool_call("request_approval", {}, "c3")]),
        AIMessage(closing_message),
    ]
    graph = build_graph(
        model=ScriptedModel(messages=iter(script)),
        registry=registry,
        contracts_dir=tmp_path,
        checkpointer=InMemorySaver(),
    )
    return graph, {"configurable": {"thread_id": "t1"}}


def test_graph_pauses_on_approval_interrupt(tmp_path, registry, synthetic_data_path):
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "done")
    result = graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)

    (intr,) = result["__interrupt__"]
    request = intr.value
    action = request["action_requests"][0]
    assert action["name"] == "approve_contract"
    assert "null_check" in action["description"]
    assert "```yaml" in action["description"]  # YAML fenced so markdown keeps indentation
    assert request["review_configs"][0]["allowed_decisions"] == ["approve", "edit", "reject"]
    assert not list(tmp_path.glob("*.yaml")), "nothing persists before approval"


def test_accept_persists_approved_contract_the_engine_runs(
    tmp_path, registry, synthetic_data_path, orders_df, monkeypatch
):
    monkeypatch.setenv("DQ_AGENT_APPROVER", "jacopo")
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "approved!")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)
    result = graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

    assert result["messages"][-1].content == "approved!"
    contract = Contract.from_yaml(tmp_path / "orders.yaml")
    assert contract.approved_at is not None
    assert contract.approved_by == "jacopo"

    # the artifact is directly executable by the deterministic engine
    results = run(contract, orders_df, registry)
    assert results[0].rule_id == "null_check"
    assert results[0].error is None


def test_reject_decision_does_not_approve(tmp_path, registry, synthetic_data_path):
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "ok, revising")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)
    result = graph.invoke(
        Command(resume={"decisions": [
            {"type": "reject", "message": "loosen the null rate"}
        ]}),
        config,
    )

    assert not list(tmp_path.glob("*.yaml")), "feedback must not persist a contract"
    assert "loosen the null rate" in result["messages"][-2].content  # fed back to agent
    assert graph.get_state(config).values["draft"] is not None  # draft survives for iteration


def test_edit_decision_persists_modified_contract(
    tmp_path, registry, synthetic_data_path, monkeypatch
):
    """An edit decision carries the owner's modified contract in edited_action.args."""
    monkeypatch.setenv("DQ_AGENT_APPROVER", "jacopo")
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "approved!")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)

    edited = graph.get_state(config).values["draft"]
    edited["rules"][0]["params"]["max_null_rate"] = 0.0  # a valid tweak by the owner
    graph.invoke(
        Command(resume={"decisions": [{
            "type": "edit",
            "edited_action": {"name": "approve_contract", "args": {"contract": edited}},
        }]}),
        config,
    )

    contract = Contract.from_yaml(tmp_path / "orders.yaml")
    assert contract.approved_at is not None
    assert contract.rules[0].params["max_null_rate"] == 0.0


def test_text_string_approve_persists(tmp_path, registry, synthetic_data_path, monkeypatch):
    """Raw-JSON fallback view: the owner types 'approve' rather than clicking a button."""
    monkeypatch.setenv("DQ_AGENT_APPROVER", "jacopo")
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "approved!")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)
    graph.invoke(Command(resume="approve"), config)

    assert (tmp_path / "orders.yaml").exists()
