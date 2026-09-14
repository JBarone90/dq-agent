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

from typing import Any, Iterator, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

DEFAULT_MODEL_ID = "eu.anthropic.claude-sonnet-5"
CACHE_TTL = "1h"
"""How long a cache entry lives. The proxy offers "5m" and "1h" at different prices.

1h is the right default here because the agent pauses at the approval gate for a
human to read a contract, which routinely runs past five minutes -- and an expired
entry means re-writing the whole transcript rather than extending it. Writes cost 2x
input at this TTL against 1.25x at 5m, but they apply to the few new tokens each turn
adds, whereas an expiry re-writes everything.

Changing this means changing the rate `_computed_cost` prices writes at, below.
"""

CACHE_POINT = {"cachePoint": {"type": "default", "ttl": CACHE_TTL}}
"""Marks how far back a request should be cached. See `DeptBedrockChat.cache_prompt`."""

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


def _invoke(
    request: dict[str, Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    stream: bool = False,
) -> Any:
    """Send one Converse request through the bedrock-proxy.

    Returns the raw `Response`, or an iterator of raw Converse stream events when
    `stream` is set. A streamed call can still fail *after* the iterator is handed
    back, so the caller must translate errors raised while iterating as well.

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


def _cache_tokens(
    usage: dict[str, Any] | None, headers: Any
) -> tuple[int | None, int | None]:
    """Prompt-cache token counts for one call, as (read, write).

    Two sources report these and neither is guaranteed: the Converse `usage` block
    (`cacheReadInputTokens` / `cacheWriteInputTokens`) and the proxy's own
    `x-cache-read-tokens` / `x-cache-write-tokens` headers. The body is preferred
    because it is Bedrock's own accounting; the headers are the fallback.

    None means nothing reported the field at all, which is worth telling apart from
    a reported zero: zero says caching is wired up and did not hit, None says we
    cannot see it. Today every call returns zero or None, because the request sets
    no cache points.
    """
    read = usage.get("cacheReadInputTokens") if usage else None
    write = usage.get("cacheWriteInputTokens") if usage else None
    if read is None:
        read = _header_int(headers, "x-cache-read-tokens")
    if write is None:
        write = _header_int(headers, "x-cache-write-tokens")
    return read, write


def _usage_metadata(
    usage: dict[str, Any] | None, headers: Any = None
) -> dict[str, Any] | None:
    """Map the Converse response `usage` block to LangChain's UsageMetadata so
    `AIMessage.usage_metadata` carries token counts for cost/usage views.

    The `usage` block is part of the Bedrock response body and is independent of the
    proxy's `show_usage` print flag, so it is read straight from `.json()`. Converse
    reports `totalTokens` itself rather than leaving it to be summed. Returns None
    when absent, so a caller can tell "unknown" apart from a genuine zero.

    Cache counts go into `input_token_details` under LangChain's own names
    (`cache_read` / `cache_creation`). Converse reports them *alongside*
    `inputTokens` rather than inside it, whereas LangChain's contract is that
    `input_tokens` is the full input with the details as subsets, so they are added
    in here. If the proxy ever folds them into `inputTokens` instead, this would
    double-count; `response_metadata['tokens_used']` (the proxy's own total, from
    `x-tokens-used`) is the cross-check that would show it.
    """
    if not usage:
        return None
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    total_tokens = usage.get("totalTokens", input_tokens + output_tokens)

    cache_read, cache_write = _cache_tokens(usage, headers)
    details: dict[str, int] = {}
    if cache_read is not None:
        details["cache_read"] = cache_read
    if cache_write is not None:
        details["cache_creation"] = cache_write
    cached = (cache_read or 0) + (cache_write or 0)

    metadata: dict[str, Any] = {
        "input_tokens": input_tokens + cached,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens + cached,
    }
    if details:
        metadata["input_token_details"] = details
    return metadata


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


_PRICING: dict[str, dict[str, float] | None] = {}
"""Per-model prices from the proxy, memoised. Prices are static per model."""


def _pricing(model_id: str) -> dict[str, float] | None:
    """The proxy's per-million-token prices for a model, or None if unavailable.

    Fetched from `get_model_info`, so these are still the proxy's prices rather than a
    table baked in here -- the same source the `x-cost` header is computed from. A
    failure is swallowed and cached as None: cost is a dev readout, and no lookup for
    it should ever take down a chat turn."""
    if model_id in _PRICING:
        return _PRICING[model_id]
    prices: dict[str, float] | None = None
    try:
        from dwutils import bedrock

        prices = bedrock.get_model_info(model_id).get("pricing")
    except Exception:  # noqa: BLE001 - a missing cost readout must not break a turn
        prices = None
    _PRICING[model_id] = prices
    return prices


def _write_rate(prices: dict[str, float]) -> float:
    """The per-million price of a cache write at the TTL we actually ask for.

    Kept next to CACHE_TTL because the two have to move together: pricing a 1-hour
    write at the 5-minute rate would quietly under-report every session."""
    return prices.get(f"cache_write_{CACHE_TTL}", prices.get("cache_write_5m", 0.0))


def _computed_cost(model_id: str, usage: dict[str, Any] | None) -> float | None:
    """Dollar cost of one call, worked out from token counts and the proxy's prices.

    Needed only for streamed turns. HTTP headers are sent before the body, so when a
    stream opens the proxy does not yet know how many output tokens there will be and
    cannot stamp `x-cost`; the token counts instead arrive in the closing `metadata`
    event. Multiplying those by the published prices recovers the number.

    Cache writes are priced at the 5-minute rate, which is what an unqualified
    `cachePoint` buys; a 1-hour entry would cost more, so this under-reports if the
    proxy is ever asked for one."""
    prices = _pricing(model_id)
    if not prices or not usage:
        return None
    charged = (
        usage.get("inputTokens", 0) * prices.get("input", 0.0)
        + usage.get("outputTokens", 0) * prices.get("output", 0.0)
        + usage.get("cacheReadInputTokens", 0) * prices.get("cache_read", 0.0)
        + usage.get("cacheWriteInputTokens", 0) * _write_rate(prices)
    )
    return charged / 1e6


def _chunk_from_event(
    event: dict[str, Any], model_id: str | None = None
) -> AIMessageChunk | None:
    """Translate one raw Converse stream event into an AIMessageChunk.

    Returns None for the events that carry nothing LangChain needs (`messageStart`,
    `contentBlockStop`), so the caller can skip them without a second lookup.

    Tool calls arrive in pieces: `contentBlockStart` names the tool and gives its id,
    then a run of `contentBlockDelta` events streams its arguments as partial JSON
    text. That is exactly LangChain's `tool_call_chunks` shape, and merging the chunks
    (`reduce(add, chunks)`) is what parses the assembled JSON into `.tool_calls` -- so
    nothing here needs to buffer or parse arguments itself. `contentBlockIndex` is the
    index that keeps parallel tool calls apart while they interleave.
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
        # Token counts arrive here rather than in headers: a streamed response carries
        # no usage headers at all, so cost is worked out from the proxy's prices.
        block = event["metadata"].get("usage")
        usage = _usage_metadata(block)
        if usage is None:
            return None
        metadata: dict[str, Any] = {}
        cost = _computed_cost(model_id, block) if model_id else None
        if cost is not None:
            metadata["cost_usd"] = cost
            metadata["cost_source"] = "computed"
        return AIMessageChunk(
            content="", usage_metadata=usage, response_metadata=metadata
        )

    return None


def _with_cache_points(request: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `request` marked up for prompt caching.

    One marker, at the end of the newest message. Because Converse assembles a
    request as tools, then system, then messages, a marker there covers the whole
    thing: tool schema, system prompt and transcript alike.

    It rolls forward on its own. Each turn resends the previous turn's request plus a
    few new messages, so last turn's stored prefix is matched automatically and this
    turn stores a slightly longer one. Nothing here tracks which turn is which.

    Reads need no marker: Bedrock matches the longest stored prefix it already holds
    and bills those tokens at the cache rate. A marker only says how much to store.

    An earlier version also marked the end of `system`, to cache the static part
    separately. Measured against the proxy that is dead weight: the static prefix is
    ~1900 tokens against a 4096-token minimum on Sonnet 5, so it can never be stored
    on its own -- and it gets cached anyway as part of this marker's longer prefix,
    whose TTL is refreshed on every turn.

    The copy is shallow but never writes through to the caller's lists, so a request
    can be built once and marked up without the markers leaking into a retry."""
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

    Bound tools are carried as a `tools` kwarg via `self.bind`, so `bind_tools`
    needs no mutable state on the instance.

    Limitations (acceptable for the scoping agent; these are the gaps to close
    before this adapter could be upstreamed into `dwutils.bedrock` as a shared
    LangChain integration):

    - **Synchronous only.** Implements `_generate`; there is no `_agenerate` /
      `_astream`, so a high-concurrency web UI blocks a worker thread per call.
    - **Streaming is supported.** `_stream` yields a chunk per Converse stream event,
      so `.stream()` and LangGraph's `stream_mode="messages"` emit tokens as they are
      produced. A streamed response carries no usage headers, so `cost_usd` and
      `tokens_used` are absent from a streamed turn and anything totalling per-call
      cost must treat such a session as incomplete; token counts still arrive, in the
      final `metadata` event.
    - **Usage and cost are surfaced.** Token counts from the response `usage` block map
      onto `AIMessage.usage_metadata`; the proxy's per-call `x-cost` / `x-tokens-used`
      response headers map onto `response_metadata["cost_usd"]` / `["tokens_used"]` (see
      `_header_cost` — cost is the proxy's, never a hardcoded price table). Account-level
      daily budget is a separate call, `dwutils.bedrock.get_usage()`, not done here.
      Prompt-cache counts are captured too, under `usage_metadata["input_token_details"]`;
      they stay at zero until the request sets cache points, which it does not yet.
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
    cache_prompt: bool = True

    def bind_tools(
        self, tools: Sequence[Any], **kwargs: Any
    ) -> Runnable[Any, BaseMessage]:
        return self.bind(tools=_to_converse_tools(tools), **kwargs)

    def _request(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Build the Converse request body. Shared by `_generate` and `_stream` so the
        two cannot drift into sending differently shaped requests."""
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
            metadata["cost_source"] = "proxy"
        tokens_used = _header_int(response.headers, "x-tokens-used")
        if tokens_used is not None:
            metadata["tokens_used"] = tokens_used

        message = AIMessage(
            content=text,
            tool_calls=tool_calls,
            usage_metadata=_usage_metadata(data.get("usage"), response.headers),
            response_metadata=metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream one turn, yielding a chunk per Converse stream event.

        Implementing this is what makes `.stream()` and LangGraph's
        `stream_mode="messages"` emit tokens as they are produced; without it
        LangChain falls back to `_generate` and hands back one chunk at the end.

        A streamed response carries no usage headers, so `cost_usd` and `tokens_used`
        are absent from a streamed turn -- token counts still arrive, in the final
        `metadata` event. Anything summing per-call cost has to treat a streamed
        session as incomplete rather than cheap.
        """
        request = self._request(messages, stop, kwargs.get("tools"))
        events = _invoke(request, max_retries=self.max_retries, stream=True)
        try:
            for event in events:
                chunk = _chunk_from_event(event, self.model_id)
                if chunk is None:
                    continue
                generation = ChatGenerationChunk(message=chunk)
                if run_manager and chunk.content:
                    run_manager.on_llm_new_token(chunk.text, chunk=generation)
                yield generation
        except RuntimeError as exc:
            # The proxy reports a mid-stream failure as an error event, which dwutils
            # raises while iterating -- after the status code is long gone. Translate it
            # here so a driver's `except BedrockProxyError` still covers streamed turns.
            raise BedrockProxyError(
                _diagnose(request, exc), getattr(exc, "status_code", None)
            ) from exc

    @property
    def _llm_type(self) -> str:
        return "dept-bedrock"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "max_tokens": self.max_tokens}
