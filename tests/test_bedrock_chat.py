"""DeptBedrockChat unit tests: message<->Anthropic translation and tool-call
parsing, with the proxy call (`_invoke`) monkeypatched so nothing touches the
network or needs `dwutils`."""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from dq_agent.agents import bedrock_chat
from dq_agent.agents.bedrock_chat import DeptBedrockChat, _to_anthropic, _to_anthropic_tools


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"sunny in {city}"


def _tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


# --- message translation -------------------------------------------------


def test_to_anthropic_splits_system_and_user():
    system, msgs = _to_anthropic([SystemMessage("be terse"), HumanMessage("hi")])
    assert system == "be terse"
    assert msgs == [{"role": "user", "content": "hi"}]


def test_to_anthropic_renders_assistant_tool_calls():
    ai = AIMessage("", tool_calls=[_tool_call("get_weather", {"city": "Paris"}, "t1")])
    _, msgs = _to_anthropic([HumanMessage("weather?"), ai])
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}}
    ]


def test_to_anthropic_coalesces_consecutive_tool_messages():
    """Parallel tool results must land in one user message (Anthropic requirement)."""
    ai = AIMessage(
        "",
        tool_calls=[
            _tool_call("get_weather", {"city": "Paris"}, "t1"),
            _tool_call("get_weather", {"city": "Rome"}, "t2"),
        ],
    )
    messages = [
        HumanMessage("weather?"),
        ai,
        ToolMessage(content="sunny in Paris", tool_call_id="t1"),
        ToolMessage(content="rainy in Rome", tool_call_id="t2"),
    ]
    _, msgs = _to_anthropic(messages)
    # one assistant turn, then a SINGLE user turn holding both tool_result blocks
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    results = msgs[-1]["content"]
    assert len(results) == 2
    assert {r["tool_use_id"] for r in results} == {"t1", "t2"}
    assert all(r["type"] == "tool_result" for r in results)


def test_to_anthropic_tools_shape():
    specs = _to_anthropic_tools([get_weather])
    assert specs[0]["name"] == "get_weather"
    assert "input_schema" in specs[0]
    assert specs[0]["input_schema"]["properties"]["city"]["type"] == "string"


# --- generation / response parsing ---------------------------------------


def test_generate_parses_text(monkeypatch):
    monkeypatch.setattr(
        bedrock_chat, "_invoke",
        lambda request: FakeResponse({"content": [{"type": "text", "text": "pong"}],
                                      "stop_reason": "end_turn"}),
    )
    result = DeptBedrockChat().invoke([HumanMessage("ping")])
    assert result.content == "pong"
    assert result.tool_calls == []
    assert result.response_metadata["stop_reason"] == "end_turn"


def test_generate_parses_tool_use(monkeypatch):
    captured = {}

    def fake_invoke(request):
        captured.update(request)
        return FakeResponse({
            "content": [{"type": "tool_use", "id": "t1", "name": "get_weather",
                         "input": {"city": "Paris"}}],
            "stop_reason": "tool_use",
        })

    monkeypatch.setattr(bedrock_chat, "_invoke", fake_invoke)

    model = DeptBedrockChat(model_id="eu.anthropic.claude-sonnet-4-6").bind_tools([get_weather])
    result = model.invoke([HumanMessage("weather in Paris?")])

    # the bound tools reached the request body in Anthropic shape
    assert captured["tools"][0]["name"] == "get_weather"
    assert captured["model_id"] == "eu.anthropic.claude-sonnet-4-6"
    # and the tool_use block became a LangChain tool call
    assert result.tool_calls == [
        {"name": "get_weather", "args": {"city": "Paris"}, "id": "t1", "type": "tool_call"}
    ]


def test_generate_includes_system_when_present(monkeypatch):
    captured = {}

    def fake_invoke(request):
        captured.update(request)
        return FakeResponse({"content": [{"type": "text", "text": "ok"}]})

    monkeypatch.setattr(bedrock_chat, "_invoke", fake_invoke)
    DeptBedrockChat().invoke([SystemMessage("be terse"), HumanMessage("hi")])
    assert captured["system"] == "be terse"
