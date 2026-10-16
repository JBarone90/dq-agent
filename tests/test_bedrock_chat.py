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


def test_generate_metadata_carries_only_the_stop_reason(monkeypatch):
    """Cost is not read from the proxy's headers: the response metadata is the stop reason."""
    monkeypatch.setattr(
        bedrock_chat, "_invoke",
        lambda request, **kw: FakeResponse(
            _converse([{"text": "ok"}]),
            headers={"x-cost": "$0.0034", "x-tokens-used": "42"},
        ),
    )
    result = DeptBedrockChat().invoke([HumanMessage("hi")])
    assert set(result.response_metadata) == {"stop_reason"}


def test_generate_includes_system_when_present(monkeypatch):
    captured = {}

    def fake_invoke(request, **kw):
        captured.update(request)
        return FakeResponse(_converse([{"text": "ok"}]))

    monkeypatch.setattr(bedrock_chat, "_invoke", fake_invoke)
    DeptBedrockChat(cache_prompt=False).invoke(
        [SystemMessage("be terse"), HumanMessage("hi")]
    )
    assert captured["system"] == [{"text": "be terse"}]


# --- prompt caching ------------------------------------------------------


def test_the_only_cache_point_is_on_the_newest_message(monkeypatch):
    """Converse orders a request tools-then-system-then-messages, so a marker on the
    last message covers all three. A second marker on `system` would be dead weight:
    the static prefix measures ~1900 tokens against a 4096-token cache minimum."""
    captured = {}

    def fake_invoke(request, **kw):
        captured.update(request)
        return FakeResponse(_converse([{"text": "ok"}]))

    monkeypatch.setattr(bedrock_chat, "_invoke", fake_invoke)
    DeptBedrockChat().bind_tools([get_weather]).invoke(
        [SystemMessage("be terse"), HumanMessage("hi")]
    )

    assert captured["messages"][-1]["content"] == [
        {"text": "hi"}, bedrock_chat.CACHE_POINT
    ]
    assert captured["system"] == [{"text": "be terse"}]
    assert bedrock_chat.CACHE_POINT not in captured["toolConfig"]["tools"]


def test_cache_points_roll_forward_with_the_conversation(monkeypatch):
    """The marker always lands on the newest message, so what this turn stores is what
    the next turn matches."""
    captured = {}

    def fake_invoke(request, **kw):
        captured.update(request)
        return FakeResponse(_converse([{"text": "ok"}]))

    monkeypatch.setattr(bedrock_chat, "_invoke", fake_invoke)
    DeptBedrockChat().invoke([
        HumanMessage("first"),
        AIMessage("reply"),
        HumanMessage("second"),
    ])

    messages = captured["messages"]
    assert messages[-1]["content"] == [{"text": "second"}, bedrock_chat.CACHE_POINT]
    assert all(
        bedrock_chat.CACHE_POINT not in m["content"] for m in messages[:-1]
    ), "only the newest message is marked"


def test_cache_prompt_off_sends_no_markers(monkeypatch):
    captured = {}

    def fake_invoke(request, **kw):
        captured.update(request)
        return FakeResponse(_converse([{"text": "ok"}]))

    monkeypatch.setattr(bedrock_chat, "_invoke", fake_invoke)
    DeptBedrockChat(cache_prompt=False).invoke(
        [SystemMessage("be terse"), HumanMessage("hi")]
    )
    assert captured["messages"][-1]["content"] == [{"text": "hi"}]


def test_cache_points_do_not_leak_into_a_rebuilt_request(monkeypatch):
    """Marking must not write through to the lists `_to_converse` built, or a retry
    would send a request with the markers doubled up."""
    request = {
        "system": [{"text": "s"}],
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    }
    marked = bedrock_chat._with_cache_points(request)
    assert marked["messages"][0]["content"] == [
        {"text": "hi"}, bedrock_chat.CACHE_POINT
    ]
    assert request["messages"][0]["content"] == [{"text": "hi"}]
    assert request["system"] == [{"text": "s"}]


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


# --- streaming -----------------------------------------------------------


def _stream_events(*events):
    """A fake `_invoke` that returns raw Converse stream events instead of a Response."""
    def fake_invoke(request, **kw):
        assert kw.get("stream") is True, "streaming must ask the proxy for a stream"
        return list(events)
    return fake_invoke


def test_stream_yields_text_chunks(monkeypatch):
    monkeypatch.setattr(bedrock_chat, "_invoke", _stream_events(
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Hel"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "lo"}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
    ))
    chunks = list(DeptBedrockChat().stream([HumanMessage("hi")]))
    assert "".join(c.content for c in chunks) == "Hello"


def test_stream_assembles_tool_calls_from_partial_json(monkeypatch):
    """Converse streams tool arguments as partial JSON text across several deltas;
    merging the chunks is what parses them back into a real tool call."""
    monkeypatch.setattr(bedrock_chat, "_invoke", _stream_events(
        {"contentBlockStart": {
            "start": {"toolUse": {"toolUseId": "t1", "name": "get_weather"}},
            "contentBlockIndex": 0,
        }},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"city":'}},
                               "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": ' "Paris"}'}},
                               "contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
    ))
    chunks = list(DeptBedrockChat().stream([HumanMessage("weather?")]))
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk
    assert merged.tool_calls == [
        {"name": "get_weather", "args": {"city": "Paris"}, "id": "t1", "type": "tool_call"}
    ]
    assert merged.response_metadata["stop_reason"] == "tool_use"


def test_stream_surfaces_usage_from_the_metadata_event(monkeypatch):
    """A streamed response carries no usage headers; the counts arrive in the final
    metadata event instead."""
    monkeypatch.setattr(bedrock_chat, "_invoke", _stream_events(
        {"contentBlockDelta": {"delta": {"text": "ok"}, "contentBlockIndex": 0}},
        {"metadata": {"usage": {"inputTokens": 12, "outputTokens": 5, "totalTokens": 17}}},
    ))
    chunks = list(DeptBedrockChat().stream([HumanMessage("hi")]))
    usage = [c.usage_metadata for c in chunks if c.usage_metadata]
    assert usage == [{"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}]


def test_stream_midstream_failure_becomes_bedrock_proxy_error(monkeypatch):
    """The proxy reports a failure that starts mid-stream as an error event, which
    dwutils raises while iterating. A driver only catches BedrockProxyError."""
    def fake_invoke(request, **kw):
        def events():
            yield {"contentBlockDelta": {"delta": {"text": "partial"}}}
            raise RuntimeError("model stopped responding")
        return events()

    monkeypatch.setattr(bedrock_chat, "_invoke", fake_invoke)
    with pytest.raises(BedrockProxyError):
        list(DeptBedrockChat().stream([HumanMessage("hi")]))


def test_stream_and_generate_send_the_same_request_shape(monkeypatch):
    """_request is shared so the two paths cannot drift; only the stream flag differs."""
    seen = []

    def fake_invoke(request, **kw):
        seen.append((request, kw.get("stream", False)))
        if kw.get("stream"):
            return [{"contentBlockDelta": {"delta": {"text": "ok"}}}]
        return FakeResponse(_converse([{"text": "ok"}]))

    monkeypatch.setattr(bedrock_chat, "_invoke", fake_invoke)
    model = DeptBedrockChat().bind_tools([get_weather])
    model.invoke([SystemMessage("be terse"), HumanMessage("hi")])
    list(model.stream([SystemMessage("be terse"), HumanMessage("hi")]))

    assert seen[0][0] == seen[1][0]
    assert [streamed for _, streamed in seen] == [False, True]


# --- cache TTL and tool binding --------------------------------------------


def test_cache_point_asks_for_the_one_hour_ttl():
    """The agent pauses at the approval gate for longer than the 5m default, so an
    unqualified cache point would expire exactly when the conversation resumes."""
    assert bedrock_chat.CACHE_POINT == {
        "cachePoint": {"type": "default", "ttl": "1h"}
    }


def test_streamed_usage_carries_cache_tokens(monkeypatch):
    monkeypatch.setattr(bedrock_chat, "_invoke", _stream_events(
        {"metadata": {"usage": {
            "inputTokens": 10, "outputTokens": 5, "totalTokens": 15,
            "cacheReadInputTokens": 900, "cacheWriteInputTokens": 40,
        }}},
    ))
    (chunk,) = [c for c in DeptBedrockChat().stream([HumanMessage("hi")]) if c.usage_metadata]
    assert chunk.usage_metadata["input_tokens"] == 950
    assert chunk.usage_metadata["input_token_details"] == {"cache_read": 900, "cache_creation": 40}
    assert not chunk.response_metadata.get("cost_usd")


def test_bind_tools_refuses_tool_choice():
    """Converse's toolChoice is not translated, so asking for one must fail loudly."""
    with pytest.raises(NotImplementedError, match="tool_choice"):
        DeptBedrockChat().bind_tools([get_weather], tool_choice="any")


# --- edges ------------------------------------------------------------------


def test_without_dwutils_the_error_says_to_inject_a_model(monkeypatch):
    """Off the work network dwutils is absent; the message must say what to do instead."""
    import sys

    monkeypatch.setitem(sys.modules, "dwutils", None)  # makes `from dwutils import bedrock` fail
    with pytest.raises(BedrockProxyError, match=r"inject a model: build_graph\(model=\.\.\.\)"):
        DeptBedrockChat().invoke([HumanMessage("hi")])


def test_an_unsupported_message_type_is_refused():
    from langchain_core.messages import ChatMessage

    with pytest.raises(TypeError, match="unsupported message type: ChatMessage"):
        bedrock_chat._to_converse([ChatMessage(role="critic", content="x")])


def test_events_that_carry_nothing_are_skipped(monkeypatch):
    monkeypatch.setattr(bedrock_chat, "_invoke", _stream_events(
        {"contentBlockStart": {"start": {}, "contentBlockIndex": 0}},  # a text block starting
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "..."}}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "ok"}, "contentBlockIndex": 0}},
        {"metadata": {}},
        {"somethingNew": {}},
    ))
    chunks = list(DeptBedrockChat().stream([HumanMessage("hi")]))
    assert [c.content for c in chunks if c.content] == ["ok"]  # LangChain adds an empty closing chunk


def test_no_cache_point_without_messages():
    request = {"model_id": "m", "messages": []}
    assert bedrock_chat._with_cache_points(request) is request


def test_streamed_tokens_reach_the_callback(monkeypatch):
    from langchain_core.callbacks import BaseCallbackHandler

    class Collect(BaseCallbackHandler):
        def __init__(self):
            self.tokens = []

        def on_llm_new_token(self, token, **kwargs):
            self.tokens.append(token)

    monkeypatch.setattr(bedrock_chat, "_invoke", _stream_events(
        {"contentBlockDelta": {"delta": {"text": "Hel"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "lo"}, "contentBlockIndex": 0}},
    ))
    handler = Collect()
    list(DeptBedrockChat().stream([HumanMessage("hi")], config={"callbacks": [handler]}))
    assert [t for t in handler.tokens if t] == ["Hel", "lo"]


@pytest.mark.parametrize("status, hint", [
    (400, "Converse shape"), (401, "BEDROCK_TOKEN"), (404, "model_id is not available"),
    (429, "throttling"), (None, "never replied"), (500, None),
])
def test_every_status_gets_its_hint(status, hint):
    class Failure(RuntimeError):
        status_code = status

    message = bedrock_chat._diagnose({"model_id": "m"}, Failure("boom"))
    assert message.startswith("bedrock-proxy call failed: boom")
    assert message.endswith("model_id: 'm'")
    if hint:
        assert hint in message
    else:
        assert len(message.splitlines()) == 2  # no hint for an unexpected status
