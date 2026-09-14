"""DeptBedrockChat unit tests: message<->Converse translation and tool-call
parsing, with the proxy call (`_invoke`) monkeypatched so nothing touches the
network or needs `dwutils`."""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from dq_agent.agents import bedrock_chat
from dq_agent.agents.bedrock_chat import (
    BedrockProxyError,
    DeptBedrockChat,
    _to_converse,
    _to_converse_tools,
)


class FakeResponse:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}
        self.text = "" if payload is _NON_JSON else str(payload)

    def json(self):
        if self._payload is _NON_JSON:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


_NON_JSON = object()  # sentinel: FakeResponse whose .json() raises, like a non-JSON body


def _converse(content, **extra):
    """A Converse response body: the reply nests under output.message.content."""
    return {"output": {"message": {"role": "assistant", "content": content}}, **extra}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"sunny in {city}"


def _tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


# --- message translation -------------------------------------------------


def test_to_converse_splits_system_and_user():
    system, msgs = _to_converse([SystemMessage("be terse"), HumanMessage("hi")])
    assert system == [{"text": "be terse"}]
    assert msgs == [{"role": "user", "content": [{"text": "hi"}]}]


def test_to_converse_renders_assistant_tool_calls():
    ai = AIMessage(content="", tool_calls=[_tool_call("get_weather", {"city": "Paris"}, "t1")])
    _, msgs = _to_converse([HumanMessage("weather?"), ai])
    assert msgs[1] == {
        "role": "assistant",
        "content": [{
            "toolUse": {"toolUseId": "t1", "name": "get_weather", "input": {"city": "Paris"}}
        }],
    }


def test_to_converse_coalesces_consecutive_tool_messages():
    """Converse requires the results of parallel tool calls in one user turn, but
    LangGraph's ToolNode emits one ToolMessage each."""
    messages = [
        HumanMessage("weather in Paris and Rome?"),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call("get_weather", {"city": "Paris"}, "t1"),
                _tool_call("get_weather", {"city": "Rome"}, "t2"),
            ],
        ),
        ToolMessage(content="sunny in Paris", tool_call_id="t1"),
        ToolMessage(content="sunny in Rome", tool_call_id="t2"),
    ]
    _, msgs = _to_converse(messages)
    assert len(msgs) == 3  # user, assistant, one merged tool-result turn
    results = msgs[2]["content"]
    assert msgs[2]["role"] == "user"
    assert [block["toolResult"]["toolUseId"] for block in results] == ["t1", "t2"]
    assert results[0]["toolResult"]["content"] == [{"text": "sunny in Paris"}]


def test_to_converse_does_not_merge_tool_result_into_plain_user_turn():
    """Every Converse user turn is a list of blocks, so 'is the content a list' cannot
    stand in for 'is this an open tool-result turn' — a tool result following ordinary
    user text must start its own message."""
    messages = [
        AIMessage(content="", tool_calls=[_tool_call("get_weather", {"city": "Rome"}, "t1")]),
        HumanMessage("actually, never mind"),
        ToolMessage(content="sunny in Rome", tool_call_id="t1"),
    ]
    _, msgs = _to_converse(messages)
    assert len(msgs) == 3
    assert msgs[1]["content"] == [{"text": "actually, never mind"}]
    assert "toolResult" in msgs[2]["content"][0]


def test_to_converse_tools_shape():
    specs = _to_converse_tools([get_weather])
    spec = specs[0]["toolSpec"]
    assert spec["name"] == "get_weather"
    assert spec["inputSchema"]["json"]["properties"]["city"]["type"] == "string"


# --- generation / response parsing ---------------------------------------


def test_generate_parses_text(monkeypatch):
    monkeypatch.setattr(
        bedrock_chat, "_invoke",
        lambda request, **kw: FakeResponse(
            _converse([{"text": "pong"}], stopReason="end_turn")
        ),
    )
    result = DeptBedrockChat().invoke([HumanMessage("ping")])
    assert result.content == "pong"
    assert result.tool_calls == []
    assert result.response_metadata["stop_reason"] == "end_turn"


def test_generate_parses_tool_use(monkeypatch):
    captured = {}

    def fake_invoke(request, **kw):
        captured.update(request)
        return FakeResponse(_converse(
            [{"toolUse": {"toolUseId": "t1", "name": "get_weather",
                          "input": {"city": "Paris"}}}],
            stopReason="tool_use",
        ))

    monkeypatch.setattr(bedrock_chat, "_invoke", fake_invoke)

    model = DeptBedrockChat(model_id="eu.anthropic.claude-sonnet-4-6").bind_tools([get_weather])
    result = model.invoke([HumanMessage("weather in Paris?")])

    # the bound tools reached the request body in Converse shape
    assert captured["toolConfig"]["tools"][0]["toolSpec"]["name"] == "get_weather"
    assert captured["model_id"] == "eu.anthropic.claude-sonnet-4-6"
    # and the toolUse block became a LangChain tool call
    assert result.tool_calls == [
        {"name": "get_weather", "args": {"city": "Paris"}, "id": "t1", "type": "tool_call"}
    ]


def test_generate_puts_decoding_params_under_inference_config(monkeypatch):
    """max_tokens and stop sequences moved into inferenceConfig under Converse."""
    captured = {}

    def fake_invoke(request, **kw):
        captured.update(request)
        return FakeResponse(_converse([{"text": "ok"}]))

    monkeypatch.setattr(bedrock_chat, "_invoke", fake_invoke)
    DeptBedrockChat(max_tokens=512).invoke([HumanMessage("hi")], stop=["STOP"])
    assert captured["inferenceConfig"] == {"maxTokens": 512, "stopSequences": ["STOP"]}
    assert "max_tokens" not in captured
    assert "anthropic_version" not in captured


def test_generate_surfaces_usage(monkeypatch):
    monkeypatch.setattr(
        bedrock_chat, "_invoke",
        lambda request, **kw: FakeResponse(_converse(
            [{"text": "ok"}],
            usage={"inputTokens": 12, "outputTokens": 5, "totalTokens": 17},
        )),
    )
    result = DeptBedrockChat().invoke([HumanMessage("hi")])
    assert result.usage_metadata == {
        "input_tokens": 12, "output_tokens": 5, "total_tokens": 17,
    }


def test_generate_usage_absent_is_none(monkeypatch):
    monkeypatch.setattr(
        bedrock_chat, "_invoke",
        lambda request, **kw: FakeResponse(_converse([{"text": "ok"}])),
    )
    result = DeptBedrockChat().invoke([HumanMessage("hi")])
    assert result.usage_metadata is None


def test_generate_captures_cost_and_tokens_from_headers(monkeypatch):
    monkeypatch.setattr(
        bedrock_chat, "_invoke",
        lambda request, **kw: FakeResponse(
            _converse([{"text": "ok"}]),
            headers={"x-cost": "$0.0034", "x-tokens-used": "42"},
        ),
    )
    result = DeptBedrockChat().invoke([HumanMessage("hi")])
    assert result.response_metadata["cost_usd"] == 0.0034
    assert result.response_metadata["tokens_used"] == 42


def test_generate_omits_cost_when_header_absent(monkeypatch):
    monkeypatch.setattr(
        bedrock_chat, "_invoke",
        lambda request, **kw: FakeResponse(_converse([{"text": "ok"}])),
    )
    result = DeptBedrockChat().invoke([HumanMessage("hi")])
    assert "cost_usd" not in result.response_metadata
    assert "tokens_used" not in result.response_metadata


def test_generate_includes_system_when_present(monkeypatch):
    captured = {}

    def fake_invoke(request, **kw):
        captured.update(request)
        return FakeResponse(_converse([{"text": "ok"}]))

    monkeypatch.setattr(bedrock_chat, "_invoke", fake_invoke)
    DeptBedrockChat().invoke([SystemMessage("be terse"), HumanMessage("hi")])
    assert captured["system"] == [{"text": "be terse"}]


def test_generate_surfaces_cache_tokens_from_usage_block(monkeypatch):
    """Converse reports cache counts alongside inputTokens, so they are added into
    input_tokens and broken out under LangChain's input_token_details names."""
    monkeypatch.setattr(
        bedrock_chat, "_invoke",
        lambda request, **kw: FakeResponse(_converse(
            [{"text": "ok"}],
            usage={"inputTokens": 100, "outputTokens": 5, "totalTokens": 105,
                   "cacheReadInputTokens": 900, "cacheWriteInputTokens": 0},
        )),
    )
    result = DeptBedrockChat().invoke([HumanMessage("hi")])
    assert result.usage_metadata["input_tokens"] == 1000
    assert result.usage_metadata["total_tokens"] == 1005
    assert result.usage_metadata["input_token_details"] == {
        "cache_read": 900, "cache_creation": 0,
    }


def test_generate_falls_back_to_cache_headers(monkeypatch):
    """The proxy stamps cache counts on the response headers too; use them when the
    Converse usage block does not carry the fields."""
    monkeypatch.setattr(
        bedrock_chat, "_invoke",
        lambda request, **kw: FakeResponse(
            _converse([{"text": "ok"}], usage={"inputTokens": 10, "outputTokens": 5}),
            headers={"x-cache-read-tokens": "40", "x-cache-write-tokens": "2"},
        ),
    )
    result = DeptBedrockChat().invoke([HumanMessage("hi")])
    assert result.usage_metadata["input_token_details"] == {
        "cache_read": 40, "cache_creation": 2,
    }
    assert result.usage_metadata["input_tokens"] == 52


def test_generate_omits_cache_details_when_unreported(monkeypatch):
    """No cache fields anywhere means "cannot see it", which must not render as a
    zero that looks like a measured cache miss."""
    monkeypatch.setattr(
        bedrock_chat, "_invoke",
        lambda request, **kw: FakeResponse(_converse(
            [{"text": "ok"}], usage={"inputTokens": 10, "outputTokens": 5},
        )),
    )
    result = DeptBedrockChat().invoke([HumanMessage("hi")])
    assert "input_token_details" not in result.usage_metadata
    assert result.usage_metadata["input_tokens"] == 10


# --- defensive error handling --------------------------------------------


def _fake_dwutils(monkeypatch, invoke):
    """Install a stand-in `dwutils` so the real `_invoke` wrapper can be exercised."""
    import sys
    import types

    class BedrockError(RuntimeError):
        def __init__(self, message, status_code=None):
            super().__init__(f"{status_code}: {message}" if status_code else message)
            self.status_code = status_code
            self.message = message

    fake = types.ModuleType("dwutils")
    fake.bedrock = types.SimpleNamespace(invoke=invoke, BedrockError=BedrockError)
    monkeypatch.setitem(sys.modules, "dwutils", fake)
    return fake.bedrock


def test_invoke_sends_a_full_request_not_a_prompt(monkeypatch):
    """`invoke` defaults to simple mode and rejects a `request=` dict outright, so the
    adapter must opt into full-request mode explicitly. Omitting this is what breaks
    every call before it reaches the network."""
    captured = {}

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return FakeResponse(_converse([{"text": "ok"}]))

    _fake_dwutils(monkeypatch, fake_invoke)
    DeptBedrockChat(max_retries=5).invoke([HumanMessage("hi")])

    assert captured["simple"] is False
    assert captured["show_usage"] is False
    assert captured["max_retries"] == 5
    assert captured["request"]["model_id"] == bedrock_chat.DEFAULT_MODEL_ID


def test_proxy_error_becomes_bedrock_proxy_error(monkeypatch):
    """A BedrockError from dwutils is restated as BedrockProxyError, keeping the
    proxy's own message and the status to branch on."""
    bedrock = None

    def boom(**kwargs):
        raise bedrock.BedrockError("Daily budget exhausted", 403)

    bedrock = _fake_dwutils(monkeypatch, boom)

    with pytest.raises(BedrockProxyError) as info:
        DeptBedrockChat(model_id="eu.anthropic.claude-sonnet-4-6").invoke(
            [HumanMessage("hi")]
        )
    assert info.value.status_code == 403
    assert "Daily budget exhausted" in str(info.value)
    assert "midnight UTC" in str(info.value)  # the 403 hint
    assert "eu.anthropic.claude-sonnet-4-6" in str(info.value)


def test_missing_proxy_env_becomes_bedrock_proxy_error(monkeypatch):
    """dwutils raises a bare ValueError when BEDROCK_PROXY_URL / BEDROCK_TOKEN are
    unset; the drivers only catch BedrockProxyError, so it must be restated."""

    def boom(**kwargs):
        raise ValueError("You don't have access to Bedrock.")

    _fake_dwutils(monkeypatch, boom)

    with pytest.raises(BedrockProxyError) as info:
        DeptBedrockChat().invoke([HumanMessage("hi")])
    assert "access to Bedrock" in str(info.value)


def test_generate_non_json_body_becomes_bedrock_proxy_error(monkeypatch):
    """A 2xx whose body is not JSON surfaces a BedrockProxyError, not a bare ValueError."""
    monkeypatch.setattr(
        bedrock_chat, "_invoke", lambda request, **kw: FakeResponse(_NON_JSON)
    )
    with pytest.raises(BedrockProxyError):
        DeptBedrockChat().invoke([HumanMessage("hi")])
