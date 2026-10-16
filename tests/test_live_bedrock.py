"""The adapter and the scoping agent against the live Bedrock proxy: run at work.

Needs `dwutils` importable and BEDROCK_PROXY_URL / BEDROCK_TOKEN set; DQ_AGENT_MODEL picks the
model (default: the adapter's). Everything here is marked `integration` and skips otherwise, so
CI never calls the proxy. Run it after a proxy or model change:

    uv run pytest -m integration -v

These replace the removed probe script's checks: a plain call, tool calling, a tool-result round
trip, streaming, prompt caching, and one end-to-end scoping session that reaches the approval gate
with a contract data-flow would accept. Each call costs a little budget; the session test costs
the most (several turns).
"""

import importlib.util
import os

import pytest

_READY = importlib.util.find_spec("dwutils") is not None and all(
    os.environ.get(name) for name in ("BEDROCK_PROXY_URL", "BEDROCK_TOKEN"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _READY, reason="needs dwutils and BEDROCK_PROXY_URL / BEDROCK_TOKEN (work environment)"),
]

pytest.importorskip("langgraph")
pytest.importorskip("great_expectations")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from dq_agent import gx_contract
from dq_agent.agents.bedrock_chat import DEFAULT_MODEL_ID, DeptBedrockChat
from dq_agent.agents.gate import pending_interrupt, resume
from dq_agent.agents.scoping import build_graph
from dq_agent.models import Contract

MODEL_ID = os.environ.get("DQ_AGENT_MODEL", DEFAULT_MODEL_ID)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is 20C and sunny in {city}."


def _model(**options):
    return DeptBedrockChat(model_id=MODEL_ID, max_tokens=512, **options)


def test_a_plain_call_answers_and_reports_usage():
    reply = _model().invoke([HumanMessage("Reply with exactly: pong")])
    assert "pong" in reply.content.lower()
    assert reply.usage_metadata["input_tokens"] > 0 and reply.usage_metadata["output_tokens"] > 0
    assert reply.response_metadata["stop_reason"] == "end_turn"


def test_the_proxy_forwards_tools_and_returns_a_tool_call():
    reply = _model().bind_tools([get_weather]).invoke([HumanMessage("What's the weather in Paris? Use the tool.")])
    (call,) = reply.tool_calls
    assert call["name"] == "get_weather" and "paris" in call["args"]["city"].lower()


def test_a_tool_result_round_trip_completes():
    model = _model().bind_tools([get_weather])
    question = HumanMessage("What's the weather in Paris? Use the tool.")
    first = model.invoke([question])
    (call,) = first.tool_calls
    final = model.invoke([question, AIMessage(content=first.content, tool_calls=first.tool_calls),
                          ToolMessage(get_weather.invoke(call["args"]), tool_call_id=call["id"])])
    assert "20" in final.content


def test_streaming_yields_text_and_a_tool_call():
    text = "".join(chunk.content for chunk in _model().stream([HumanMessage("Count from 1 to 5.")])
                   if isinstance(chunk.content, str))
    assert "5" in text

    merged = None
    for chunk in _model().bind_tools([get_weather]).stream([HumanMessage("Weather in Rome? Use the tool.")]):
        merged = chunk if merged is None else merged + chunk
    assert merged.tool_calls[0]["name"] == "get_weather"


def test_a_repeated_prefix_is_read_from_the_cache():
    """The second identical call should read the first's cached prefix. A failure here means
    caching is off for the account: run with DQ_AGENT_PROMPT_CACHE=0 (see BEDROCK_PROXY.md)."""
    system = SystemMessage("You are a careful data quality assistant. " * 400)  # over any cache minimum
    messages = [system, HumanMessage("Reply with exactly: one")]
    model = _model()
    model.invoke(messages)
    second = model.invoke(messages).usage_metadata
    assert second.get("input_token_details", {}).get("cache_read", 0) > 0, second


def test_a_scoping_session_reaches_the_gate_with_a_valid_contract(tmp_path, monkeypatch):
    """One real session on the synthetic orders file: the agent profiles it, proposes a contract
    data-flow would accept, and approving it writes the file. The wording varies run to run, so
    only the outcome is checked."""
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

    result = graph.invoke(resume({"type": "approve"}), config)
    assert result.get("contract_path")
    gx_contract.load_contract(tmp_path / "synthetic.orders.yaml")  # what data-flow loads
