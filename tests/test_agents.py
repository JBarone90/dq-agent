"""Scoping agent tests: tools are exercised directly; the graph (routing, approval
interrupt, contract persistence) is exercised with a scripted fake model — no LLM
calls anywhere in the suite."""

import pytest

pytest.importorskip("langgraph")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from dq_agent.agents.scoping import ProposedRule, _make_tools, build_graph
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
    request = intr.value[0]
    assert request["action_request"]["action"] == "approve_contract"
    assert "null_check" in request["description"]
    assert not list(tmp_path.glob("*.yaml")), "nothing persists before approval"


def test_accept_persists_approved_contract_the_engine_runs(
    tmp_path, registry, synthetic_data_path, orders_df, monkeypatch
):
    monkeypatch.setenv("DQ_AGENT_APPROVER", "jacopo")
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "approved!")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)
    result = graph.invoke(Command(resume=[{"type": "accept", "args": None}]), config)

    assert result["messages"][-1].content == "approved!"
    contract = Contract.from_yaml(tmp_path / "orders.yaml")
    assert contract.approved_at is not None
    assert contract.approved_by == "jacopo"

    # the artifact is directly executable by the deterministic engine
    results = run(contract, orders_df, registry)
    assert results[0].rule_id == "null_check"
    assert results[0].error is None


def test_feedback_response_does_not_approve(tmp_path, registry, synthetic_data_path):
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "ok, revising")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)
    result = graph.invoke(
        Command(resume=[{"type": "response", "args": "loosen the null rate"}]), config
    )

    assert not list(tmp_path.glob("*.yaml")), "feedback must not persist a contract"
    assert "loosen the null rate" in result["messages"][-2].content  # fed back to agent
    assert graph.get_state(config).values["draft"] is not None  # draft survives for iteration
