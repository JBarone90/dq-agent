"""LangChain chat model backed by the in-house bedrock-proxy (`dwutils.bedrock`).

The work environment has no internet and no AWS SigV4 path — Bedrock is reached
through an internal HTTP proxy (`dwutils.bedrock.invoke`) that forwards an
Anthropic-on-Bedrock request body verbatim and returns the raw response. This
module wraps that single call in a `BaseChatModel` so the scoping agent's
provider-agnostic plumbing (`build_graph(model=...)`, `bind_tools`) works
unchanged — the LLM stays a swappable config detail, exactly as in CLAUDE.md.

The agent depends on tool calling, so the translation here is the load-bearing
part: LangChain messages -> Anthropic `messages`/`system`/`tools`, and the
response's `content` blocks -> text + `tool_calls`.

`dwutils` is the internal package and is not importable outside the work
environment; the import is deferred into `_invoke` so this module loads (and the
unit tests run) anywhere. Tests monkeypatch `_invoke` to avoid the proxy.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

DEFAULT_MODEL_ID = "eu.anthropic.claude-sonnet-4-6"
ANTHROPIC_VERSION = "bedrock-2023-05-31"


def _invoke(request: dict[str, Any]) -> Any:
    """Send one request through the bedrock-proxy and return the raw `Response`.

    Imported lazily so this module loads without `dwutils` (e.g. on a dev laptop
    or in CI); the unit tests monkeypatch this function.
    """
    from dwutils import bedrock  # internal package, only present at work

    return bedrock.invoke(request=request, show_usage=False)


def _to_anthropic(messages: Sequence[BaseMessage]) -> tuple[str, list[dict[str, Any]]]:
    """Translate LangChain messages into (system_text, anthropic messages array).

    Consecutive `ToolMessage`s are coalesced into a single user turn: Anthropic
    requires every `tool_result` answering a parallel tool call to live in one
    message, but LangGraph's ToolNode emits one `ToolMessage` per call.
    """
    system = ""
    msgs: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system += message.content + "\n"
        elif isinstance(message, HumanMessage):
            msgs.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["name"],
                    "input": call["args"],
                })
            msgs.append({"role": "assistant", "content": content})
        elif isinstance(message, ToolMessage):
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": str(message.content),
            }
            # merge into the open tool-result turn if the previous message is one
            if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                msgs[-1]["content"].append(block)
            else:
                msgs.append({"role": "user", "content": [block]})
        else:
            raise TypeError(f"unsupported message type: {type(message).__name__}")
    return system.strip(), msgs


def _to_anthropic_tools(tools: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert LangChain tools to Anthropic tool specs (name/description/input_schema)."""
    specs = []
    for tool in tools:
        fn = convert_to_openai_tool(tool)["function"]
        specs.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn["parameters"],
        })
    return specs


def _usage_metadata(usage: dict[str, Any] | None) -> dict[str, int] | None:
    """Map the Anthropic response `usage` block to LangChain's UsageMetadata so
    `AIMessage.usage_metadata` carries token counts for cost/usage views.

    The `usage` block is part of the Bedrock response body and is independent of the
    proxy's `show_usage` print flag, so it is read straight from `.json()`. Returns
    None when absent, so a caller can tell "unknown" apart from a genuine zero."""
    if not usage:
        return None
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _cost_usd(data: dict[str, Any]) -> float | None:
    """Per-call cost when the proxy enriches the response with it, else None.

    The raw Anthropic body carries tokens only — `dwutils.bedrock` is the pricing
    source (this deliberately does NOT hardcode a price table, which would drift and
    ignore the org's negotiated Bedrock rates). The exact field the proxy uses is not
    yet pinned, so this checks the likely shapes; returns None when absent so the UI
    shows nothing rather than a wrong number. Confirm with
    `inspect.getsource(dwutils.bedrock.invoke)` and tighten to the real key."""
    usage = data.get("usage") or {}
    for value in (data.get("cost"), data.get("cost_usd"),
                  usage.get("cost"), usage.get("cost_usd")):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


class DeptBedrockChat(BaseChatModel):
    """Chat model that calls Bedrock (Claude) through the internal bedrock-proxy.

    Bound tools are carried as a `tools` kwarg via `self.bind`, so `bind_tools`
    needs no mutable state on the instance.

    Limitations (acceptable for the scoping agent; these are the gaps to close
    before this adapter could be upstreamed into `dwutils.bedrock` as a shared
    LangChain integration):

    - **Synchronous only.** Implements `_generate`; there is no `_agenerate` /
      `_astream`, so a high-concurrency web UI blocks a worker thread per call.
    - **No streaming.** `_stream` is unimplemented — a full response is returned at
      once, so a UI cannot render tokens as they are produced.
    - **Usage and (when the proxy provides it) cost are surfaced.** Token counts from
      the response `usage` block map onto `AIMessage.usage_metadata`; a per-call cost
      is read from the response when present and stashed in
      `response_metadata["cost_usd"]` (see `_cost_usd` — no hardcoded price table;
      `dwutils.bedrock` is the pricing source). Cache-read/-write token fields are not
      captured.
    - **Minimal error handling.** A proxy/HTTP error or a malformed response body
      surfaces raw, not as a typed, retryable error.
    - **Fixed decoding params.** `max_tokens` defaults to 10000; temperature / top_p
      are not plumbed through.
    """

    model_id: str = DEFAULT_MODEL_ID
    max_tokens: int = 10000

    def bind_tools(
        self, tools: Sequence[Any], **kwargs: Any
    ) -> Runnable[Any, BaseMessage]:
        return self.bind(tools=_to_anthropic_tools(tools), **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, anthropic_messages = _to_anthropic(messages)
        request: dict[str, Any] = {
            "model_id": self.model_id,
            "anthropic_version": ANTHROPIC_VERSION,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }
        if system:
            request["system"] = system
        if kwargs.get("tools"):
            request["tools"] = kwargs["tools"]
        if stop:
            request["stop_sequences"] = stop

        data = _invoke(request).json()

        text = ""
        tool_calls = []
        for block in data.get("content", []):
            kind = block.get("type")
            if kind == "text":
                text += block["text"]
            elif kind == "tool_use":
                tool_calls.append({
                    "name": block["name"],
                    "args": block["input"],
                    "id": block["id"],
                    "type": "tool_call",
                })

        metadata: dict[str, Any] = {"stop_reason": data.get("stop_reason")}
        cost = _cost_usd(data)
        if cost is not None:
            metadata["cost_usd"] = cost

        message = AIMessage(
            content=text,
            tool_calls=tool_calls,
            usage_metadata=_usage_metadata(data.get("usage")),
            response_metadata=metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "dept-bedrock"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "max_tokens": self.max_tokens}
