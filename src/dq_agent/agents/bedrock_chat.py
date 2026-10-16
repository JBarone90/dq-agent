"""LangChain chat model backed by the in-house bedrock-proxy (`dwutils.bedrock`).

The work environment reaches Bedrock only through an internal HTTP proxy that forwards a
Bedrock Converse request body. This module wraps that call in a `BaseChatModel` so the
scoping agent's `build_graph(model=...)` and `bind_tools` work unchanged. The load-bearing
part is the translation between LangChain messages and Converse messages, system blocks
and `toolConfig`, and back to text and `tool_calls`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

DEFAULT_MODEL_ID = "eu.anthropic.claude-sonnet-5"
CACHE_TTL = "1h"
# 1h rather than 5m: the agent waits at the approval gate while a human reads the
# contract, which routinely outlasts five minutes, and an expired entry re-writes the
# whole transcript.
CACHE_POINT = {"cachePoint": {"type": "default", "ttl": CACHE_TTL}}

DEFAULT_MAX_RETRIES = 3  # attempts per call, including the first; dwutils does the backoff


class BedrockProxyError(RuntimeError):
    """A failed bedrock-proxy call, so drivers can catch one type without importing
    `dwutils` (absent off-network). `status_code` is None when the proxy never replied."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


_STATUS_HINTS = {
    400: "The proxy rejected the request body (check the Converse shape, `toolConfig` included).",
    401: "BEDROCK_TOKEN is missing or expired.",
    403: (
        "Either the token is not authorised, or the daily budget is spent — that one "
        "clears at midnight UTC. `dwutils.bedrock.get_usage()` says which."
    ),
    404: "The model_id is not available on the proxy.",
    429: (
        "Bedrock is throttling and the retries ran out. Try again, or raise "
        "`max_retries` on the model."
    ),
}


def _diagnose(request: dict[str, Any], exc: Exception) -> str:
    """An actionable message for a failed proxy call: the proxy's own message, plus a hint
    for the statuses where the fix is specific to this adapter's configuration."""
    status = getattr(exc, "status_code", None)
    lines = [f"bedrock-proxy call failed: {exc}"]
    hint = _STATUS_HINTS.get(status) if isinstance(status, int) else None
    if hint:
        lines.append(hint)
    elif status is None:
        lines.append(
            "The proxy never replied — check BEDROCK_PROXY_URL is set and reachable."
        )
    lines.append(f"model_id: {request.get('model_id')!r}")
    return "\n".join(lines)


def _invoke(
    request: dict[str, Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    stream: bool = False,
) -> Any:
    """Send one Converse request through the proxy: the raw `Response`, or an iterator of
    stream events. A stream can fail while being iterated, so callers translate errors
    there too. `dwutils` is imported lazily so the module loads without it; tests
    monkeypatch this function."""
    try:
        from dwutils import bedrock  # internal package, only present at work
    except ImportError as exc:
        raise BedrockProxyError(
            "dwutils is not importable — the bedrock proxy is only available in the work "
            "environment. To run elsewhere, inject a model: build_graph(model=...)."
        ) from exc

    try:
        return bedrock.invoke(
            request=request,
            simple=False,  # full-request mode; without it `request=` is rejected
            stream=stream,
            show_usage=False,
            max_retries=max_retries,
        )
    except RuntimeError as exc:
        # dwutils' BedrockError subclasses RuntimeError and carries `.status_code`.
        raise BedrockProxyError(
            _diagnose(request, exc), getattr(exc, "status_code", None)
        ) from exc
    except ValueError as exc:
        # Raised when BEDROCK_PROXY_URL / BEDROCK_TOKEN are unset, and on a request
        # dwutils considers malformed. Both are configuration problems, not transport.
        raise BedrockProxyError(f"bedrock-proxy call rejected: {exc}") from exc


def _to_converse(
    messages: Sequence[BaseMessage],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Translate LangChain messages into (system blocks, Converse messages array).

    Consecutive `ToolMessage`s become one user turn: Converse wants every `toolResult`
    for a parallel tool call in a single message, but ToolNode emits one per call.
    """
    system: list[dict[str, Any]] = []
    msgs: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system.append({"text": message.content})
        elif isinstance(message, HumanMessage):
            msgs.append({"role": "user", "content": [{"text": message.content}]})
        elif isinstance(message, AIMessage):
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"text": message.content})
            for call in message.tool_calls:
                content.append({
                    "toolUse": {
                        "toolUseId": call["id"],
                        "name": call["name"],
                        "input": call["args"],
                    }
                })
            msgs.append({"role": "assistant", "content": content})
        elif isinstance(message, ToolMessage):
            block = {
                "toolResult": {
                    "toolUseId": message.tool_call_id,
                    "content": [{"text": str(message.content)}],
                }
            }
            if msgs and _is_tool_result_turn(msgs[-1]):
                msgs[-1]["content"].append(block)
            else:
                msgs.append({"role": "user", "content": [block]})
        else:
            raise TypeError(f"unsupported message type: {type(message).__name__}")
    return system, msgs


def _is_tool_result_turn(message: dict[str, Any]) -> bool:
    """True if `message` is a user turn already carrying tool results."""
    return message["role"] == "user" and any(
        "toolResult" in block for block in message["content"]
    )


def _to_converse_tools(tools: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert LangChain tools to Converse tool specs (`toolSpec` with a JSON schema)."""
    specs = []
    for tool in tools:
        fn = convert_to_openai_tool(tool)["function"]
        specs.append({
            "toolSpec": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "inputSchema": {"json": fn["parameters"]},
            }
        })
    return specs


def _usage_metadata(usage: dict[str, Any] | None) -> UsageMetadata | None:
    """Map the Converse `usage` block to LangChain's UsageMetadata, or None when absent.

    Converse reports cache reads/writes alongside `inputTokens`, whereas LangChain's
    `input_tokens` is the full input, so they are added in.
    """
    if not usage:
        return None
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    total_tokens = usage.get("totalTokens", input_tokens + output_tokens)

    cache_read = usage.get("cacheReadInputTokens")
    cache_write = usage.get("cacheWriteInputTokens")
    details: dict[str, int] = {}
    if cache_read is not None:
        details["cache_read"] = cache_read
    if cache_write is not None:
        details["cache_creation"] = cache_write
    cached = (cache_read or 0) + (cache_write or 0)

    metadata: UsageMetadata = {
        "input_tokens": input_tokens + cached,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens + cached,
    }
    if details:
        metadata["input_token_details"] = cast(InputTokenDetails, details)
    return metadata


def _chunk_from_event(event: dict[str, Any]) -> AIMessageChunk | None:
    """Translate one raw Converse stream event into an AIMessageChunk, or None for events
    that carry nothing LangChain needs.

    A tool call arrives as a `contentBlockStart` naming the tool, then `contentBlockDelta`
    events streaming its arguments as partial JSON. That is LangChain's `tool_call_chunks`
    shape; merging the chunks parses the JSON, and `contentBlockIndex` keeps parallel
    calls apart.
    """
    if "contentBlockStart" in event:
        block = event["contentBlockStart"]
        use = block.get("start", {}).get("toolUse")
        if not use:
            return None
        return AIMessageChunk(
            content="",
            tool_call_chunks=[{
                "name": use.get("name"),
                "args": "",
                "id": use.get("toolUseId"),
                "index": block.get("contentBlockIndex", 0),
                "type": "tool_call_chunk",
            }],
        )

    if "contentBlockDelta" in event:
        block = event["contentBlockDelta"]
        delta = block.get("delta", {})
        if "text" in delta:
            return AIMessageChunk(content=delta["text"])
        if "toolUse" in delta:
            return AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": None,
                    "args": delta["toolUse"].get("input", ""),
                    "id": None,
                    "index": block.get("contentBlockIndex", 0),
                    "type": "tool_call_chunk",
                }],
            )
        return None

    if "messageStop" in event:
        return AIMessageChunk(
            content="",
            response_metadata={"stop_reason": event["messageStop"].get("stopReason")},
        )

    if "metadata" in event:
        usage = _usage_metadata(event["metadata"].get("usage"))
        return AIMessageChunk(content="", usage_metadata=usage) if usage else None

    return None


def _with_cache_points(request: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `request` with one cache marker at the end of the newest message.

    Converse assembles a request as tools, system, then messages, so that marker covers
    the whole prefix. It rolls forward on its own: each turn resends the previous request
    plus a few messages, so the stored prefix is matched and a slightly longer one stored.
    The caller's lists are never written through, so a retry never sees the markers.
    """
    messages = request.get("messages")
    if not messages:
        return request
    last = messages[-1]
    return {
        **request,
        "messages": [
            *messages[:-1],
            {**last, "content": [*last["content"], CACHE_POINT]},
        ],
    }


class DeptBedrockChat(BaseChatModel):
    """Chat model that calls Bedrock (Claude) through the internal bedrock-proxy.

    Synchronous only (no `_agenerate`/`_astream`); `max_tokens` is fixed and temperature
    is not plumbed through. Retries belong to `dwutils.bedrock`; a failure that survives
    them is re-raised as `BedrockProxyError`. Token counts, including cache reads and
    writes, land on `AIMessage.usage_metadata`.
    """

    model_id: str = DEFAULT_MODEL_ID
    max_tokens: int = 10000
    max_retries: int = DEFAULT_MAX_RETRIES
    cache_prompt: bool = True

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        if tool_choice is not None:
            # not translated to Converse's toolChoice: refuse rather than silently ignore it
            raise NotImplementedError("DeptBedrockChat does not support tool_choice")
        return cast(Runnable[LanguageModelInput, AIMessage],
                    self.bind(tools=_to_converse_tools(tools), **kwargs))

    def _request(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """The Converse request body, shared by `_generate` and `_stream`."""
        system, converse_messages = _to_converse(messages)
        inference: dict[str, Any] = {"maxTokens": self.max_tokens}
        if stop:
            inference["stopSequences"] = stop
        request: dict[str, Any] = {
            "model_id": self.model_id,
            "messages": converse_messages,
            "inferenceConfig": inference,
        }
        if system:
            request["system"] = system
        if tools:
            request["toolConfig"] = {"tools": tools}
        return _with_cache_points(request) if self.cache_prompt else request

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        request = self._request(messages, stop, kwargs.get("tools"))
        response = _invoke(request, max_retries=self.max_retries)
        try:
            data = response.json()
        except ValueError as exc:  # 2xx but a non-JSON body — surface it, don't mask
            raise BedrockProxyError(
                "bedrock-proxy returned a response that is not JSON. The call succeeded at "
                "the transport level but the body could not be parsed as a Converse "
                f"response. First bytes: {str(getattr(response, 'text', ''))[:200]!r}"
            ) from exc

        text = ""
        tool_calls = []
        # Converse nests the reply under output.message and tags blocks by key, not "type".
        content = data.get("output", {}).get("message", {}).get("content", [])
        for block in content:
            if "text" in block:
                text += block["text"]
            elif "toolUse" in block:
                use = block["toolUse"]
                tool_calls.append({
                    "name": use["name"],
                    "args": use["input"],
                    "id": use["toolUseId"],
                    "type": "tool_call",
                })

        message = AIMessage(
            content=text,
            tool_calls=tool_calls,
            usage_metadata=_usage_metadata(data.get("usage")),
            response_metadata={"stop_reason": data.get("stopReason")},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream one turn, yielding a chunk per Converse stream event (this is what makes
        `.stream()` and LangGraph's `stream_mode="messages"` emit tokens)."""
        request = self._request(messages, stop, kwargs.get("tools"))
        events = _invoke(request, max_retries=self.max_retries, stream=True)
        try:
            for event in events:
                chunk = _chunk_from_event(event)
                if chunk is None:
                    continue
                generation = ChatGenerationChunk(message=chunk)
                if run_manager and chunk.content:
                    run_manager.on_llm_new_token(chunk.text, chunk=generation)
                yield generation
        except RuntimeError as exc:
            # a mid-stream failure is raised by dwutils while iterating; translate it
            # so `except BedrockProxyError` covers streamed turns too
            raise BedrockProxyError(
                _diagnose(request, exc), getattr(exc, "status_code", None)
            ) from exc

    @property
    def _llm_type(self) -> str:
        return "dept-bedrock"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "max_tokens": self.max_tokens}
