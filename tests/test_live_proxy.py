"""The default model and the scoping agent against the live bedrock-proxy: run at work.

Needs ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN set; DQ_AGENT_MODEL picks the model (default:
`llm.DEFAULT_MODEL_ID`). Everything here is marked `integration` and skips otherwise, so CI never
calls the proxy. Run it after a proxy, model or `langchain-anthropic` change:

    uv run pytest -m integration -v

It checks what the agent relies on: a plain call, tool calling, a tool-result round trip,
streaming, prompt caching, and one end-to-end scoping session that reaches the approval gate with
a contract data-flow would accept. Each call costs a little budget; the session test costs the
most (several turns).
"""

import os

import pytest

_READY = all(os.environ.get(name) for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _READY, reason="needs ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN (work environment)"),
]

pytest.importorskip("langgraph")
pytest.importorskip("langchain_anthropic")
pytest.importorskip("great_expectations")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from dq_agent import gx_contract
from dq_agent.agents.gate import pending_interrupt, resume
from dq_agent.agents.llm import DEFAULT_MODEL_ID, default_model
from dq_agent.agents.scoping import build_graph
from dq_agent.models import Contract

MODEL_ID = os.environ.get("DQ_AGENT_MODEL", DEFAULT_MODEL_ID)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is 20C and sunny in {city}."


def _model(**options):
    return default_model(MODEL_ID, **options)


def test_a_plain_call_answers_and_reports_usage():
    reply = _model().invoke([HumanMessage("Reply with exactly: pong")])
    assert "pong" in reply.text.lower()
    assert reply.usage_metadata["input_tokens"] > 0 and reply.usage_metadata["output_tokens"] > 0
    assert reply.response_metadata["stop_reason"] == "end_turn"


def test_the_proxy_forwards_tools_and_returns_a_tool_call():
    reply = _model().bind_tools([get_weather]).invoke([HumanMessage("What's the weather in Paris? Use the tool.")])
    (call,) = reply.tool_calls
    assert call["name"] == "get_weather" and "paris" in call["args"]["city"].lower()


def test_a_tool_result_round_trip_completes():
    """The first reply goes back unchanged, thinking blocks included, as the graph does."""
    model = _model().bind_tools([get_weather])
    question = HumanMessage("What's the weather in Paris? Use the tool.")
    first = model.invoke([question])
    (call,) = first.tool_calls
    final = model.invoke([question, first,
                          ToolMessage(get_weather.invoke(call["args"]), tool_call_id=call["id"])])
    assert "20" in final.text


def test_streaming_yields_text_and_a_tool_call():
    text = "".join(chunk.text for chunk in _model().stream([HumanMessage("Count from 1 to 5.")]))
    assert "5" in text
    merged = None
    for chunk in _model().bind_tools([get_weather]).stream([HumanMessage("Weather in Rome? Use the tool.")]):
        merged = chunk if merged is None else merged + chunk
    assert merged.tool_calls[0]["name"] == "get_weather"


def test_a_repeated_prefix_is_read_from_the_cache():
    """The second identical call should read the first's cached prefix. A failure here means the
    proxy ignores top-level `cache_control`: run with DQ_AGENT_PROMPT_CACHE=0 until it is fixed."""
    system = SystemMessage("You are a careful data quality assistant. " * 800)  # over any cache minimum
    messages = [system, HumanMessage("Reply with exactly: one")]
    model = _model()
    model.invoke(messages)
    second = model.invoke(messages).usage_metadata
    assert second.get("input_token_details", {}).get("cache_read", 0) > 0, second


def test_a_scoping_session_reaches_the_gate_with_a_valid_contract(tmp_path, monkeypatch):
    """One real session on the synthetic orders file: the agent profiles it, proposes a contract
    data-flow would accept, and approving it writes the file. The wording varies run to run, so
    only the outcome is checked, plus that later turns read the transcript from the cache."""
    monkeypatch.setenv("DQ_AGENT_APPROVER", "pytest")
    graph = build_graph(contracts_dir=tmp_path, proposals_dir=tmp_path, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "live-session"}}
    message = (
        "Please scope the dataset at data/synthetic/orders.csv. It will run against the Postgres table "
        "synthetic.orders. Every order needs a customer_id, order_id is unique, amount is never negative. "
        "Use your own judgement for everything else and severity, and propose the contract straight away "
        "without asking me further questions."
    )
    result = graph.invoke({"messages": [HumanMessage(message)]}, config)
    for _ in range(4):  # the agent may still ask something; keep telling it to go ahead
        if pending_interrupt(result):
            break
        result = graph.invoke({"messages": [HumanMessage("Go ahead with your suggestions and propose it.")]}, config)
    gate = pending_interrupt(result)
    assert gate, "the agent never reached the approval gate"

    draft = Contract.model_validate(gate["action_requests"][0]["args"]["contract"])
    assert draft.problems() == []
    checks = {(e.expectation_type, e.kwargs.get("column")) for e in draft.suite.expectations}
    assert ("expect_column_values_to_not_be_null", "customer_id") in checks

    usage = [m.usage_metadata for m in result["messages"] if isinstance(m, AIMessage) and m.usage_metadata]
    assert any(u.get("input_token_details", {}).get("cache_read", 0) > 0 for u in usage), usage

    result = graph.invoke(resume({"type": "approve"}), config)
    assert result.get("contract_path")
    gx_contract.load_contract(tmp_path / "synthetic.orders.yaml")  # what data-flow loads
