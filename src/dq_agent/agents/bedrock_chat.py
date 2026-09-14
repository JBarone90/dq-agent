"""LangChain chat model backed by the in-house bedrock-proxy (`dwutils.bedrock`).

The work environment has no internet and no AWS SigV4 path — Bedrock is reached
through an internal HTTP proxy (`dwutils.bedrock.invoke`) that forwards a
**Bedrock Converse** request body and returns the raw response. This module wraps
that single call in a `BaseChatModel` so the scoping agent's provider-agnostic
plumbing (`build_graph(model=...)`, `bind_tools`) works unchanged — the LLM stays
a swappable config detail, exactly as in CLAUDE.md.

The agent depends on tool calling, so the translation here is the load-bearing
part: LangChain messages -> Converse `messages`/`system`/`toolConfig`, and the
response's `output.message.content` blocks -> text + `tool_calls`.

Wire format (Converse, not the Anthropic Messages shape this adapter first
targeted): `system` is a list of blocks, every message's `content` is a list of
blocks with no `"type"` discriminator, decoding params live under
`inferenceConfig`, tools under `toolConfig`, and the reply arrives nested under
`output.message.content` with a camelCase `stopReason` and `usage`.

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

DEFAULT_MODEL_ID = "eu.anthropic.claude-sonnet-5"
DEFAULT_MAX_RETRIES = 3
"""Attempts per call, including the first — `dwutils.bedrock` does the backoff."""


class BedrockProxyError(RuntimeError):
    """A call to the bedrock-proxy failed in a way the caller should act on.

    `dwutils.bedrock` raises its own `BedrockError` carrying the proxy's message and
    the HTTP status it came back with. This restates that as one exception type the
    drivers (`scripts/scoping_cli.py`, `app/scoping_app.py`) can catch without
    importing `dwutils`, which is not present off-network. `status_code` is carried
    through so a caller can branch on the cause (403 = budget spent, 400 = the
    request body was rejected); it is None when the proxy never replied.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


_STATUS_HINTS = {
    400: (
        "The proxy rejected the request body. Run "
        "`uv run python scripts/explore_bedrock.py` to check the Converse shape, "
        "`toolConfig` included, against the live proxy."
    ),
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
    """Build an actionable message for a failed proxy call.

    The proxy's own message leads, since `dwutils.bedrock` now decodes error bodies
    properly and reports the real HTTP status. We only add a hint for the statuses
    where the fix is specific to this adapter's configuration.
    """
    status = getattr(exc, "status_code", None)
    lines = [f"bedrock-proxy call failed: {exc}"]
    hint = _STATUS_HINTS.get(status)
    if hint:
        lines.append(hint)
    elif status is None:
        lines.append(
            "The proxy never replied — check BEDROCK_PROXY_URL is set and reachable."
        )
    lines.append(f"model_id: {request.get('model_id')!r}")
    return "\n".join(lines)


def _invoke(request: dict[str, Any], *, max_retries: int = DEFAULT_MAX_RETRIES) -> Any:
    """Send one Converse request through the bedrock-proxy, returning the raw `Response`.

    `simple=False` is what selects full-request mode; without it `invoke` treats a
    `request=` dict as a mistake and raises before any call is made. Retries for
    throttling and dropped connections happen inside `dwutils.bedrock`.

    `dwutils` is imported lazily so this module loads without it (e.g. on a dev laptop
    or in CI); the unit tests monkeypatch this function.
    """
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
            simple=False,
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

    Consecutive `ToolMessage`s are coalesced into a single user turn: Converse
    requires every `toolResult` answering a parallel tool call to live in one
    message, but LangGraph's ToolNode emits one `ToolMessage` per call.
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
            # Merge into the open tool-result turn if the previous message is one.
            # Every user turn is a list of blocks under Converse, so the test has to
            # be "does it hold tool results", not "is the content a list".
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


def _usage_metadata(usage: dict[str, Any] | None) -> dict[str, int] | None:
    """Map the Converse response `usage` block to LangChain's UsageMetadata so
    `AIMessage.usage_metadata` carries token counts for cost/usage views.

    The `usage` block is part of the Bedrock response body and is independent of the
    proxy's `show_usage` print flag, so it is read straight from `.json()`. Converse
    reports `totalTokens` itself rather than leaving it to be summed. Returns None when
    absent, so a caller can tell "unknown" apart from a genuine zero."""
    if not usage:
        return None
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("totalTokens", input_tokens + output_tokens),
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
      once, so a UI cannot render tokens as they are produced. The proxy does support
      it (`bedrock.invoke(..., stream=True)`), so this is a gap in the adapter rather
      than in the transport.
    - **Usage and cost are surfaced.** Token counts from the response `usage` block map
      onto `AIMessage.usage_metadata`; the proxy's per-call `x-cost` / `x-tokens-used`
      response headers map onto `response_metadata["cost_usd"]` / `["tokens_used"]` (see
      `_header_cost` — cost is the proxy's, never a hardcoded price table). Account-level
      daily budget is a separate call, `dwutils.bedrock.get_usage()`, not done here.
      Cache-read/-write token fields are not captured.
    - **Retries belong to the proxy.** `dwutils.bedrock` retries throttling and dropped
      connections with backoff, honouring `Retry-After`; `max_retries` below is passed
      through to it. A failure that survives that is re-raised as `BedrockProxyError`
      carrying the proxy's own message and `status_code`.
    - **Fixed decoding params.** `max_tokens` defaults to 10000; temperature / top_p
      are not plumbed through.
    """

    model_id: str = DEFAULT_MODEL_ID
    max_tokens: int = 10000
    max_retries: int = DEFAULT_MAX_RETRIES

    def bind_tools(
        self, tools: Sequence[Any], **kwargs: Any
    ) -> Runnable[Any, BaseMessage]:
        return self.bind(tools=_to_converse_tools(tools), **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
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
        if kwargs.get("tools"):
            request["toolConfig"] = {"tools": kwargs["tools"]}

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
        # Converse nests the reply under output.message, and tags each block by its
        # key rather than by a "type" field.
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

        # Per-call cost/tokens live in the proxy response headers, not the body.
        metadata: dict[str, Any] = {"stop_reason": data.get("stopReason")}
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
