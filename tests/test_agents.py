"""Scoping agent tests: tools are exercised directly; the graph (routing, approval
interrupt, contract persistence) is exercised with a scripted fake model — no LLM
calls anywhere in the suite."""

import json

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("great_expectations")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import dq_agent.connectors as connectors
from dq_agent.agents import bedrock_chat
from dq_agent.agents.bedrock_chat import DeptBedrockChat
from dq_agent.agents.gate import pending_interrupt, saved_result
from dq_agent.agents.scoping import ProposedExpectation, _make_tools, build_graph
from dq_agent.connectors import ProfilingLoad
# the loader data-flow applies, in dq-agent's copy of its rules: what the agent saves must load there
from dq_agent.gx_contract import load_contract
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


def test_profile_dataset_refuses_files_outside_the_data_dir(registry, synthetic_data_path, tmp_path):
    """The path comes from the LLM, so only the data directory can be read."""
    outside = tmp_path / "secrets.csv"
    outside.write_text("a\n1\n")
    tools = {t.name: t for t in _make_tools(registry, data_dir=synthetic_data_path)}
    command = tools["profile_dataset"].invoke(_tool_call("profile_dataset", {"path": str(outside)}))
    assert "only files under the data directory" in command.update["messages"][0].content
    assert "profile" not in command.update


def test_profile_dataset_reports_an_unreadable_file_as_a_tool_error(registry, tmp_path):
    (tmp_path / "broken.parquet").write_bytes(b"not parquet")
    tools = {t.name: t for t in _make_tools(registry, data_dir=tmp_path)}
    command = tools["profile_dataset"].invoke(
        _tool_call("profile_dataset", {"path": str(tmp_path / "broken.parquet")})
    )
    assert "could not read" in command.update["messages"][0].content
    assert "profile" not in command.update


def test_list_rules_filters_by_tag(tools):
    catalogue = tools["list_rules"].invoke({"tags": ["completeness"]})
    assert "expect_column_values_to_not_be_null" in catalogue
    assert "expect_column_values_to_be_unique" not in catalogue


def test_list_rules_unfiltered_lists_registry(tools, registry):
    catalogue = tools["list_rules"].invoke({})
    for rule_id in registry.rule_ids:
        assert rule_id in catalogue


def test_propose_contract_requires_profile(tools):
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"expectations": [{"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "order_id"}}], "table": "synthetic.orders",
         "state": {}},
    ))
    assert "profile the dataset first" in command.update["messages"][0].content


def test_propose_contract_rejects_unknown_rule(tools, synthetic_data_path):
    profiled = tools["profile_dataset"].invoke(
        _tool_call("profile_dataset", {"path": str(synthetic_data_path / "orders.csv")})
    )
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"expectations": [{"expectation_type": "does_not_exist", "kwargs": {}}],
         "table": "synthetic.orders", "state": {"profile": profiled.update["profile"]}},
    ))
    assert "'does_not_exist' is not an allowed" in command.update["messages"][0].content
    assert "draft" not in command.update


def test_propose_contract_records_unapproved_draft(tools, synthetic_data_path):
    profiled = tools["profile_dataset"].invoke(
        _tool_call("profile_dataset", {"path": str(synthetic_data_path / "orders.csv")})
    )
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"expectations": [{
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "order_id"},
            "meta": {"notes": {"format": "string", "content": "every order has an id"}},
        }], "table": "synthetic.orders", "state": {"profile": profiled.update["profile"]}},
    ))
    draft = Contract.model_validate(command.update["draft"])
    assert draft.approval is None
    assert draft.dataset.name == "synthetic.orders"
    assert draft.suite.expectations[0].meta.notes.content == "every order has an id"


def test_propose_contract_keeps_severity(tools, synthetic_data_path):
    profiled = tools["profile_dataset"].invoke(
        _tool_call("profile_dataset", {"path": str(synthetic_data_path / "orders.csv")})
    )
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"expectations": [{
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {"column": "amount", "min_value": 0},
            "meta": {"severity": "warning"},
        }], "table": "synthetic.orders", "state": {"profile": profiled.update["profile"]}},
    ))
    draft = Contract.model_validate(command.update["draft"])
    assert draft.suite.expectations[0].severity == "warning"


def test_propose_contract_from_file_needs_target_table(tools, synthetic_data_path):
    profiled = tools["profile_dataset"].invoke(
        _tool_call("profile_dataset", {"path": str(synthetic_data_path / "orders.csv")})
    )
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"expectations": [{"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "order_id"}}], "state": {"profile": profiled.update["profile"]}},
    ))
    assert "which Postgres table" in command.update["messages"][0].content
    assert "draft" not in command.update


def test_propose_contract_defaults_target_to_profiled_table(tools, orders_df, monkeypatch):
    _mock_pg(monkeypatch, orders_df, sampled=False, estimated_rows=20)
    profiled = tools["profile_table"].invoke(
        _tool_call("profile_table", {"table": "public.orders"})
    )
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"expectations": [{"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "order_id"}}],
         "state": {"profile": profiled.update["profile"], "source_table": "public.orders"}},
    ))
    assert Contract.model_validate(command.update["draft"]).dataset.name == "public.orders"


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
        {"expectations": [{"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "amount", "min_value": 0, "max_value": 5000}}],
         "state": {"profile": profiled.update["profile"], "source_table": "public.orders"}},
    ))
    assert spy == ["amount"]  # confirmed exactly the referenced column, once
    content = command.update["messages"][0].content
    assert "existing rows would violate" in content
    assert "99999" in content  # the real max, above the proposed max_value


def _propose_on_sampled_table(tools, orders_df, monkeypatch, expectation, spy):
    _mock_pg(monkeypatch, orders_df, sampled=True, estimated_rows=5_000_000,
             bounds=(-200, 99999), bounds_spy=spy)
    profiled = tools["profile_table"].invoke(_tool_call("profile_table", {"table": "public.orders"}))
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"expectations": [expectation],
         "state": {"profile": profiled.update["profile"], "source_table": "public.orders"}},
    ))
    return command.update["messages"][0].content


def test_propose_contract_confirms_a_max_bound_against_the_real_max(tools, orders_df, monkeypatch):
    spy = []
    content = _propose_on_sampled_table(tools, orders_df, monkeypatch, {
        "expectation_type": "expect_column_max_to_be_between",
        "kwargs": {"column": "amount", "max_value": 5000},
    }, spy)
    assert spy == ["amount"]
    assert "actual max 99999 is above the proposed max_value 5000" in content


def test_propose_contract_confirms_a_min_bound_against_the_real_min(tools, orders_df, monkeypatch):
    content = _propose_on_sampled_table(tools, orders_df, monkeypatch, {
        "expectation_type": "expect_column_min_to_be_between",
        "kwargs": {"column": "amount", "min_value": 0},
    }, [])
    assert "actual min -200 is below the proposed min_value 0" in content


def test_propose_contract_skips_bounds_for_a_freshness_parameter(tools, orders_df, monkeypatch):
    """A {"$PARAMETER": ...} bound is evaluated by GX at run time: there is nothing to
    query, and comparing it to a date must not surface as a 'could not confirm' note."""
    spy = []
    content = _propose_on_sampled_table(tools, orders_df, monkeypatch, {
        "expectation_type": "expect_column_min_to_be_between",
        "kwargs": {"column": "created_at", "min_value": {"$PARAMETER": "now() - timedelta(days=30)"}},
    }, spy)
    assert spy == []
    assert "could not confirm" not in content


def test_propose_contract_skips_bounds_when_not_sampled(tools, orders_df, monkeypatch):
    spy = []
    _mock_pg(monkeypatch, orders_df, sampled=False, estimated_rows=20,
             bounds=(-200, 99999), bounds_spy=spy)
    profiled = tools["profile_table"].invoke(
        _tool_call("profile_table", {"table": "public.orders"})
    )
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"expectations": [{"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "amount", "min_value": 0, "max_value": 5000}}],
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
        {"expectations": [{"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "order_id"}}], "table": "synthetic.orders",
         "state": {"profile": profiled.update["profile"], "source_table": "public.orders"}},
    ))
    assert spy == []  # no range expectation -> nothing to confirm


# --- proposed expectation kwargs coercion --------------------------------

NOT_NULL = "expect_column_values_to_not_be_null"


def test_proposed_expectation_coerces_json_string_kwargs():
    """Some models serialize kwargs as a JSON string; the validator parses it so the
    first tool call validates instead of needing a retry."""
    e = ProposedExpectation(expectation_type=NOT_NULL, kwargs='{"column": "email"}')
    assert e.kwargs == {"column": "email"}


def test_proposed_expectation_keeps_dict_kwargs_untouched():
    e = ProposedExpectation(expectation_type=NOT_NULL, kwargs={"column": "email"})
    assert e.kwargs == {"column": "email"}


def test_propose_contract_accepts_string_kwargs(tools, synthetic_data_path):
    """End to end through the tool: string kwargs no longer fail validation."""
    profiled = tools["profile_dataset"].invoke(
        _tool_call("profile_dataset", {"path": str(synthetic_data_path / "orders.csv")})
    )
    command = tools["propose_contract"].invoke(_tool_call(
        "propose_contract",
        {"expectations": [{"expectation_type": "expect_column_values_to_be_unique",
                           "kwargs": '{"column": "order_id"}'}],
         "table": "synthetic.orders", "state": {"profile": profiled.update["profile"]}},
    ))
    draft = Contract.model_validate(command.update["draft"])
    assert draft.suite.expectations[0].kwargs == {"column": "order_id"}


# --- graph: approval gate ------------------------------------------------


def _scoping_graph(tmp_path, registry, synthetic_data_path, closing_message):
    csv = str(synthetic_data_path / "orders.csv")
    script = [
        AIMessage("", tool_calls=[_tool_call("profile_dataset", {"path": csv}, "c1")]),
        AIMessage("", tool_calls=[_tool_call(
            "propose_contract",
            {"expectations": [{"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "order_id"}}], "table": "synthetic.orders"},
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
    assert "`order_id` must never be empty" in action["description"]
    assert "```yaml" in action["description"]  # YAML fenced so markdown keeps indentation
    assert request["review_configs"][0]["allowed_decisions"] == ["approve", "edit", "reject"]
    assert not list(tmp_path.glob("*.yaml")), "nothing persists before approval"


def test_reopened_thread_resumes_at_the_gate(tmp_path, registry, synthetic_data_path):
    """A driver reopening a saved thread finds the pending gate rather than starting a new turn,
    which would leave the approval tool call unanswered."""
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "done")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)

    saved = saved_result(graph, config)
    assert pending_interrupt(saved)["action_requests"][0]["name"] == "approve_contract"
    assert saved_result(graph, {"configurable": {"thread_id": "never-used"}}) is None


def test_accept_persists_contract_data_flow_loads(
    tmp_path, registry, synthetic_data_path, monkeypatch
):
    monkeypatch.setenv("DQ_AGENT_APPROVER", "jacopo")
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "approved!")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)
    result = graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

    assert result["messages"][-1].content == "approved!"
    # saved under the data-flow name and accepted by the data-flow loader as is
    loaded = load_contract(tmp_path / "synthetic.orders.yaml")
    assert loaded["approval"]["approved_by"] == "jacopo"
    assert loaded["suite"]["expectations"][0]["kwargs"] == {"column": "order_id"}


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
    edited["suite"]["expectations"][0]["kwargs"]["mostly"] = 0.9  # a valid tweak by the owner
    graph.invoke(
        Command(resume={"decisions": [{
            "type": "edit",
            "edited_action": {"name": "approve_contract", "args": {"contract": edited}},
        }]}),
        config,
    )

    contract = Contract.from_yaml(tmp_path / "synthetic.orders.yaml")
    assert contract.approval is not None
    assert contract.suite.expectations[0].kwargs["mostly"] == 0.9


def test_text_string_approve_persists(tmp_path, registry, synthetic_data_path, monkeypatch):
    """Raw-JSON fallback view: the owner types 'approve' rather than clicking a button."""
    monkeypatch.setenv("DQ_AGENT_APPROVER", "jacopo")
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "approved!")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)
    graph.invoke(Command(resume="approve"), config)

    assert (tmp_path / "synthetic.orders.yaml").exists()


def test_bare_decision_resume_persists(tmp_path, registry, synthetic_data_path, monkeypatch):
    monkeypatch.setenv("DQ_AGENT_APPROVER", "jacopo")
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "approved!")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)
    graph.invoke(Command(resume=[{"type": "approve"}]), config)

    assert (tmp_path / "synthetic.orders.yaml").exists()


def test_typed_feedback_resume_does_not_approve(tmp_path, registry, synthetic_data_path):
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "ok, revising")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)
    result = graph.invoke(Command(resume="make amount a warning"), config)

    assert not list(tmp_path.glob("*.yaml"))
    assert "make amount a warning" in result["messages"][-2].content


def test_approving_over_an_existing_contract_says_so(
    tmp_path, registry, synthetic_data_path, monkeypatch
):
    monkeypatch.setenv("DQ_AGENT_APPROVER", "jacopo")
    (tmp_path / "synthetic.orders.yaml").write_text("previous: contract\n")
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "approved!")
    paused = graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)
    assert "Approving replaces the existing contract" in (
        paused["__interrupt__"][0].value["action_requests"][0]["description"]
    )

    result = graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    assert "replacing the previous contract" in result["messages"][-2].content
    assert load_contract(tmp_path / "synthetic.orders.yaml")["approval"]["approved_by"] == "jacopo"


def test_edit_for_a_different_table_is_refused(tmp_path, registry, synthetic_data_path):
    """The owner reviewed a card for the draft's table; an edit cannot retarget it."""
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "ok")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)

    edited = graph.get_state(config).values["draft"]
    edited["dataset"]["table"] = "customers"
    edited["suite"]["name"] = "synthetic.customers"
    result = graph.invoke(
        Command(resume={"decisions": [{
            "type": "edit",
            "edited_action": {"name": "approve_contract", "args": {"contract": edited}},
        }]}),
        config,
    )
    assert not list(tmp_path.glob("*.yaml"))
    assert "targets a different table" in result["messages"][-2].content


def _converse_tool_stream(name, args, call_id):
    """Raw Converse stream events for one turn that calls a single tool.

    Written at the wire level on purpose: this drives the real DeptBedrockChat._stream
    rather than a LangChain fake, because the streamed tool-call path (partial JSON
    arguments reassembled across deltas) is ours to get right, not LangChain's."""
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {
            "start": {"toolUse": {"toolUseId": call_id, "name": name}},
            "contentBlockIndex": 0,
        }},
        {"contentBlockDelta": {
            "delta": {"toolUse": {"input": json.dumps(args)}},
            "contentBlockIndex": 0,
        }},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]


def test_streamed_run_still_pauses_on_the_approval_interrupt(
    tmp_path, registry, synthetic_data_path, monkeypatch
):
    """A driver that streams must still find the approval gate.

    `graph.stream` yields message chunks rather than the result dict `invoke` returns,
    so `__interrupt__` never reaches the caller: the pause shows up only in the
    persisted snapshot. That is why the Streamlit driver reloads state from the
    checkpointer after streaming instead of trusting a return value.
    """
    csv = str(synthetic_data_path / "orders.csv")
    turns = iter([
        _converse_tool_stream("profile_dataset", {"path": csv}, "c1"),
        _converse_tool_stream(
            "propose_contract",
            {"expectations": [{"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "order_id"}}], "table": "synthetic.orders"},
            "c2",
        ),
        _converse_tool_stream("request_approval", {}, "c3"),
    ])
    monkeypatch.setattr(bedrock_chat, "_invoke", lambda request, **kw: next(turns))

    graph = build_graph(
        model=DeptBedrockChat(),
        registry=registry,
        contracts_dir=tmp_path,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "stream-1"}}
    list(graph.stream(
        {"messages": [HumanMessage("scope orders.csv")]},
        config,
        stream_mode="messages",
    ))

    snapshot = graph.get_state(config)
    interrupts = [i for task in snapshot.tasks for i in getattr(task, "interrupts", ())]
    assert interrupts, "streamed run did not surface the approval interrupt"
    action = interrupts[0].value["action_requests"][0]
    assert action["name"] == "approve_contract"
    assert "```yaml" in action["description"]
    assert not list(tmp_path.glob("*.yaml")), "nothing persists before approval"


# --- creative mode --------------------------------------------------------


def test_list_unlisted_expectations_shows_gx_docs(tools):
    catalogue = tools["list_unlisted_expectations"].invoke({})
    assert "expect_column_mean_to_be_between" in catalogue
    assert "expect_column_values_to_be_unique" not in catalogue  # already allowed
    detail = tools["list_unlisted_expectations"].invoke(
        {"expectation_type": "expect_column_mean_to_be_between"}
    )
    assert "min_value" in detail and "Postgres: supported" in detail


def test_propose_new_expectation_writes_proposal_not_draft(registry, tmp_path):
    tools = {t.name: t for t in _make_tools(registry, tmp_path)}
    message = tools["propose_new_expectation"].invoke({
        "expectation_type": "expect_column_mean_to_be_between",
        "rationale": "the average order is between 10 and 200",
        "example_kwargs": {"column": "amount", "min_value": 10, "max_value": 200},
        "example_pass": "a day of typical orders",
        "example_fail": "a day where a pricing bug zeroes most amounts",
        "table": "synthetic.orders",
        "state": {},
    })
    assert "not part of the contract" in message
    assert len(list(tmp_path.glob("*.yaml"))) == 1  # a proposal file, nothing else


def test_propose_new_expectation_rejects_allowed_one(tools, tmp_path):
    message = tools["propose_new_expectation"].invoke({
        "expectation_type": "expect_column_values_to_be_unique",
        "rationale": "x", "example_kwargs": {"column": "id"},
        "example_pass": "1", "example_fail": "1, 1",
        "table": "synthetic.orders", "state": {},
    })
    assert "already allowed" in message


# --- edges of the tools and the gate --------------------------------------


def test_proposed_kwargs_that_are_not_json_are_refused():
    with pytest.raises(ValueError):
        ProposedExpectation(expectation_type="expect_column_values_to_be_unique", kwargs="{not json")


def test_profile_dataset_refuses_a_directory_and_an_unsupported_type(registry, tmp_path):
    (tmp_path / "notes.txt").write_text("x\n", encoding="utf-8")
    tools = {t.name: t for t in _make_tools(registry, data_dir=tmp_path)}
    profile = tools["profile_dataset"]
    assert "no file at" in profile.invoke(
        _tool_call("profile_dataset", {"path": str(tmp_path)})).update["messages"][0].content
    assert "unsupported file type '.txt'" in profile.invoke(
        _tool_call("profile_dataset", {"path": str(tmp_path / "notes.txt")})).update["messages"][0].content


def test_profile_dataset_reports_an_unreadable_file(registry, tmp_path):
    (tmp_path / "broken.parquet").write_bytes(b"not parquet")
    tools = {t.name: t for t in _make_tools(registry, data_dir=tmp_path)}
    content = tools["profile_dataset"].invoke(
        _tool_call("profile_dataset", {"path": str(tmp_path / "broken.parquet")})).update["messages"][0].content
    assert content.startswith("error: could not read")


def test_list_unlisted_expectations_refuses_an_allowed_one(tools):
    assert "already allowed" in tools["list_unlisted_expectations"].invoke(
        {"expectation_type": "expect_column_values_to_be_unique"})


def test_propose_new_expectation_needs_a_table(tools):
    result = tools["propose_new_expectation"].invoke({
        "expectation_type": "expect_column_mean_to_be_between", "rationale": "r",
        "example_kwargs": {"column": "amount", "min_value": 1}, "example_pass": "p", "example_fail": "f",
        "state": {"profile": {}, "source_table": None},
    })
    assert "which Postgres table" in result


def test_propose_contract_needs_a_table_for_a_file(tools, synthetic_data_path):
    profiled = tools["profile_dataset"].invoke(
        _tool_call("profile_dataset", {"path": str(synthetic_data_path / "orders.csv")}))
    command = tools["propose_contract"].invoke(_tool_call("propose_contract", {
        "expectations": [{"expectation_type": "expect_column_values_to_be_unique", "kwargs": {"column": "order_id"}}],
        "state": {"profile": profiled.update["profile"], "source_table": None},
    }))
    assert "which Postgres table" in command.update["messages"][0].content


def _propose_range_on_sampled_table(tools, profiled):
    return tools["propose_contract"].invoke(_tool_call("propose_contract", {
        "expectations": [{"expectation_type": "expect_column_values_to_be_between",
                          "kwargs": {"column": "amount", "min_value": 0}}],
        "state": {"profile": profiled.update["profile"], "source_table": "public.orders"},
    })).update["messages"][0].content


def test_bounds_note_when_the_dsn_is_gone(tools, orders_df, monkeypatch):
    _mock_pg(monkeypatch, orders_df, sampled=True, estimated_rows=5_000_000)
    profiled = tools["profile_table"].invoke(_tool_call("profile_table", {"table": "public.orders"}))

    def no_dsn(*args, **kwargs):
        raise KeyError("environment variable 'DATABASE_DSN__datasets_1' is not set")

    monkeypatch.setattr(connectors, "resolve_dsn", no_dsn)
    assert "could not confirm range bounds against the table" in _propose_range_on_sampled_table(tools, profiled)


def test_bounds_note_when_the_bounds_query_fails(tools, orders_df, monkeypatch):
    _mock_pg(monkeypatch, orders_df, sampled=True, estimated_rows=5_000_000)

    def broken(uri, table, column):
        raise RuntimeError("relation does not exist")

    monkeypatch.setattr(connectors, "column_bounds", broken)
    profiled = tools["profile_table"].invoke(_tool_call("profile_table", {"table": "public.orders"}))
    assert "range on 'amount': could not confirm bounds (relation does not exist)" in (
        _propose_range_on_sampled_table(tools, profiled))


def _gate_graph(tmp_path, registry, script):
    return build_graph(model=ScriptedModel(messages=iter(script)), registry=registry,
                       contracts_dir=tmp_path, checkpointer=InMemorySaver())


def test_approval_without_a_draft_asks_for_one(tmp_path, registry):
    graph = _gate_graph(tmp_path, registry, [
        AIMessage("", tool_calls=[_tool_call("request_approval", {}, "c1")]), AIMessage("ok")])
    result = graph.invoke({"messages": [HumanMessage("approve it")]}, {"configurable": {"thread_id": "t"}})
    assert "no draft contract" in result["messages"][-2].content
    assert not list(tmp_path.glob("*.yaml"))


def test_tool_calls_bundled_with_the_approval_are_refused(tmp_path, registry, synthetic_data_path):
    csv = str(synthetic_data_path / "orders.csv")
    graph = _gate_graph(tmp_path, registry, [
        AIMessage("", tool_calls=[_tool_call("request_approval", {}, "c1"),
                                  _tool_call("profile_dataset", {"path": csv}, "c2")]),
        AIMessage("ok"),
    ])
    result = graph.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "t"}})
    replies = {m.tool_call_id: m.content for m in result["messages"] if m.type == "tool"}
    assert replies["c2"] == "skipped: resolve the approval request first"
    assert "no draft contract" in replies["c1"]


@pytest.mark.parametrize("edited, message", [
    ("not a contract", "edited contract is malformed"),
    ({"dataset": {"schema": "synthetic", "table": "orders"},
      "suite": {"name": "synthetic.orders", "expectations": [
          {"expectation_type": "expect_column_values_to_be_unique", "kwargs": {"column": "x", "evil": 1}}]}},
     "edited contract is invalid"),
])
def test_a_bad_edit_goes_back_to_the_agent(tmp_path, registry, synthetic_data_path, edited, message):
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "revising")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)
    result = graph.invoke(Command(resume={"decisions": [{"type": "edit", "edited_action": {
        "name": "approve_contract", "args": {"contract": edited}}}]}), config)
    assert message in result["messages"][-2].content
    assert not list(tmp_path.glob("*.yaml"))


def test_the_approver_named_in_the_decision_wins(tmp_path, registry, synthetic_data_path, monkeypatch):
    monkeypatch.setenv("DQ_AGENT_APPROVER", "someone-else")
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "done")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)
    graph.invoke(Command(resume={"decisions": [{"type": "approve", "args": {"approved_by": "owner@dept"}}]}), config)
    assert load_contract(tmp_path / "synthetic.orders.yaml")["approval"]["approved_by"] == "owner@dept"


def test_a_reject_without_a_message_says_so(tmp_path, registry, synthetic_data_path):
    graph, config = _scoping_graph(tmp_path, registry, synthetic_data_path, "ok")
    graph.invoke({"messages": [HumanMessage("scope orders.csv")]}, config)
    result = graph.invoke(Command(resume={"decisions": [{"type": "reject"}]}), config)
    assert result["messages"][-2].content == "owner did not approve; feedback: none given"


def test_build_graph_defaults_to_the_bedrock_model_and_repo_registry(monkeypatch):
    """No proxy call happens at build time, so the default graph compiles anywhere."""
    monkeypatch.setenv("DQ_AGENT_MODEL", "eu.anthropic.test-model")
    monkeypatch.setenv("DQ_AGENT_PROMPT_CACHE", "0")
    built = {}
    real = DeptBedrockChat.bind_tools

    def spy(self, tools, **kwargs):
        built["model"] = self
        return real(self, tools, **kwargs)

    monkeypatch.setattr(DeptBedrockChat, "bind_tools", spy)
    graph = build_graph()
    assert set(graph.nodes) >= {"agent", "tools", "approval"}
    assert built["model"].model_id == "eu.anthropic.test-model"
    assert built["model"].cache_prompt is False
