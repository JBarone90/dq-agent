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


class BedrockProxyError(RuntimeError):
    """A call to the bedrock-proxy failed in a way the caller should act on.

    Exists to replace the confusing exception dwutils surfaces on a proxy error:
    `dwutils.bedrock.invoke` calls `raise_for_status()` and then, in its handler, does
    `raise RuntimeError(error.response.json())` — but a proxy error body is usually not
    JSON (an empty body, or an HTML 401/403/502 page), so `.json()` itself raises a
    `JSONDecodeError` that *masks* the real HTTP status. We catch that here and raise a
    message that names the likely cause instead."""


def _diagnose(request: dict[str, Any], exc: Exception) -> str:
    """Build an actionable message for a failed proxy call. Names the usual suspects so
    a reader does not have to decode dwutils' masked `JSONDecodeError`."""
    return (
        f"bedrock-proxy call failed ({type(exc).__name__}: {exc}). "
        "The proxy returned an error before a usable response; dwutils can mask the real "
        "HTTP status while formatting it. Most likely one of:\n"
        "  - BEDROCK_TOKEN missing or expired (401/403);\n"
        f"  - model_id {request.get('model_id')!r} not available on the proxy (run "
        "`scripts/explore_bedrock.py` to list models);\n"
        "  - BEDROCK_PROXY_URL wrong or unreachable (timeout/connection error).\n"
        "Run `uv run python scripts/explore_bedrock.py` to see the live proxy status."
    )


def _invoke(request: dict[str, Any]) -> Any:
    """Send one request through the bedrock-proxy and return the raw `Response`.

    `dwutils` is imported lazily so this module loads without it (e.g. on a dev laptop
    or in CI); the unit tests monkeypatch this function. Proxy/transport failures are
    re-raised as `BedrockProxyError` with a diagnostic message — see that class for why
    the raw exception is unhelpful.
    """
    try:
        from dwutils import bedrock  # internal package, only present at work
    except ImportError as exc:
        raise BedrockProxyError(
            "dwutils is not importable — the bedrock proxy is only available in the work "
            "environment. To run elsewhere, inject a model: build_graph(model=...)."
        ) from exc

    # JSONDecodeError (the masked case) subclasses RequestException; RuntimeError is what
    # dwutils raises when the error body *is* JSON. Both mean the same: proxy call failed.
    from requests.exceptions import RequestException

    try:
        return bedrock.invoke(request=request, show_usage=False)
    except (RequestException, RuntimeError) as exc:
        raise BedrockProxyError(_diagnose(request, exc)) from exc


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


def _header_cost(headers: Any) -> float | None:
    """Per-call cost in USD from the proxy's `x-cost` response header, else None.

    The bedrock-proxy stamps `x-cost` (and `x-tokens-used`) on every response; the
    `show_usage` flag only controls whether dwutils *prints* them, so we read the
    header ourselves regardless of it — and the cost comes from the proxy's pricing,
    never a hardcoded table. Returns None if the header is missing or unparseable, so
    a cost view shows nothing rather than a wrong number."""
    raw = headers.get("x-cost") if headers else None
    if raw is None:
        return None
    try:
        return float(str(raw).lstrip("$").strip())
    except (TypeError, ValueError):
        return None


def _header_int(headers: Any, name: str) -> int | None:
    """Parse an integer response header (e.g. `x-tokens-used`), None if absent/bad."""
    raw = headers.get(name) if headers else None
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
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
    - **Usage and cost are surfaced.** Token counts from the response `usage` block map
      onto `AIMessage.usage_metadata`; the proxy's per-call `x-cost` / `x-tokens-used`
      response headers map onto `response_metadata["cost_usd"]` / `["tokens_used"]` (see
      `_header_cost` — cost is the proxy's, never a hardcoded price table). Account-level
      daily budget is a separate call, `dwutils.bedrock.get_usage()`, not done here.
      Cache-read/-write token fields are not captured.
    - **Error handling is diagnostic, not retryable.** A proxy/transport failure or a
      non-JSON body is re-raised as `BedrockProxyError` with a message naming the likely
      cause (token, model_id, proxy URL) — but there is no automatic retry/backoff.
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

        response = _invoke(request)
        try:
            data = response.json()
        except ValueError as exc:  # 2xx but a non-JSON body — surface it, don't mask
            raise BedrockProxyError(
                "bedrock-proxy returned a response that is not JSON. The call succeeded at "
                "the transport level but the body could not be parsed as an Anthropic "
                f"response. First bytes: {str(getattr(response, 'text', ''))[:200]!r}"
            ) from exc

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

        # Per-call cost/tokens live in the proxy response headers, not the body.
        metadata: dict[str, Any] = {"stop_reason": data.get("stop_reason")}
        cost = _header_cost(response.headers)
        if cost is not None:
            metadata["cost_usd"] = cost
        tokens_used = _header_int(response.headers, "x-tokens-used")
        if tokens_used is not None:
            metadata["tokens_used"] = tokens_used

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
