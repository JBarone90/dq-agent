"""Run-at-work probe for the bedrock-proxy integration. NOT production code.

This is a one-time confidence check for the air-gapped work environment. It is
not imported by the agent or engine and never runs in production — it only
verifies that (1) the proxy is reachable, (2) it forwards a Converse `toolConfig`
and returns `toolUse` blocks, and (3) DeptBedrockChat round-trips tool calls end
to end.

Proxy wire format note: the proxy speaks the **Bedrock Converse** shape — a
`system` list, content blocks without a `"type"` key, `inferenceConfig.maxTokens`,
and a reply nested under `output.message.content`. `dwutils.bedrock` documents
`system`/`messages`/`inferenceConfig` but shows no tools example; step 3 confirmed
at work that the proxy accepts the standard Converse `toolConfig`, and stays here
as the regression check for it.

Prereqs (work environment only):
    - `dwutils` importable
    - BEDROCK_PROXY_URL and BEDROCK_TOKEN set (see .env.example)
    - `uv sync --extra agents`

Run:
    uv run python scripts/explore_bedrock.py
"""

from __future__ import annotations

import json
import os
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

MODEL_ID = os.environ.get("DQ_AGENT_MODEL", "eu.anthropic.claude-sonnet-5")


def _hr(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is 20C and sunny in {city}."


def step_0_env() -> None:
    _hr("0. Environment check")
    missing = [v for v in ("BEDROCK_PROXY_URL", "BEDROCK_TOKEN") if not os.environ.get(v)]
    if missing:
        print(f"MISSING env vars: {', '.join(missing)} — set them (see .env.example).")
        sys.exit(1)
    print("BEDROCK_PROXY_URL and BEDROCK_TOKEN are set.")
    print(f"DQ_AGENT_MODEL = {MODEL_ID}")


def step_1_models() -> None:
    _hr("1. Available models (bedrock.get_available_models)")
    from dwutils import bedrock

    models = bedrock.get_available_models()
    for model in models:
        print(f"  - {model}")
    if MODEL_ID not in models:
        print(f"\nWARNING: {MODEL_ID!r} is not in the list above — pick one that is.")
        return
    info = bedrock.get_model_info(MODEL_ID)
    print(f"\n  limits/pricing for {MODEL_ID}:")
    print(f"    {json.dumps(info, indent=4)[1:-1].strip()}")


def step_2_text() -> None:
    _hr("2. Plain text call (bedrock.invoke, simple mode)")
    from dwutils import bedrock

    result = bedrock.invoke(
        model_id=MODEL_ID, prompt="Reply with exactly: pong", show_usage=False
    )
    print(f"  model_response_string: {result['model_response_string']!r}")


def step_3_raw_tool_probe() -> bool:
    """The go/no-go test: does the proxy forward `toolConfig` and return a toolUse block?

    Confirmed passing at work. Tools are the one part of the Converse body
    `dwutils.bedrock` does not document, so a 400 here means the tool spelling has
    drifted rather than the proxy being broken; the message printed is the proxy's own.
    """
    _hr("3. Raw tool-use probe (bedrock.invoke with a Converse toolConfig)")
    from dwutils import bedrock

    request = {
        "model_id": MODEL_ID,
        "system": [{"text": "Use the provided tools to answer."}],
        "messages": [{"role": "user", "content": [{"text": "What's the weather in Paris?"}]}],
        "inferenceConfig": {"maxTokens": 1024},
        "toolConfig": {
            "tools": [{
                "toolSpec": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        }
                    },
                }
            }]
        },
    }

    try:
        data = bedrock.invoke(request=request, simple=False, show_usage=False).json()
    except bedrock.BedrockError as error:
        print(f"  FAIL: the proxy rejected the request ({error.status_code}).")
        print(f"  {error.message}")
        if error.status_code == 400:
            print(
                "\n  A 400 points at the request shape, most likely `toolConfig`. Check "
                "what the proxy expects for tools and update `_to_converse_tools` in "
                "src/dq_agent/agents/bedrock_chat.py to match."
            )
        return False

    print(f"  stopReason: {data.get('stopReason')}")
    content = data.get("output", {}).get("message", {}).get("content", [])
    tool_uses = [block["toolUse"] for block in content if "toolUse" in block]
    if tool_uses:
        for block in tool_uses:
            print(f"  toolUse -> {block['name']}({block['input']})")
        print("\n  PASS: the proxy forwards toolConfig and returns toolUse blocks.")
        return True
    print("\n  FAIL: no toolUse block. content was:")
    print(f"  {content}")
    print("  The proxy may be stripping tools — the scoping agent cannot run on it.")
    return False


def step_4_adapter_tool_call() -> None:
    _hr("4. Adapter end-to-end (DeptBedrockChat.bind_tools -> tool_calls)")
    from dq_agent.agents.bedrock_chat import DeptBedrockChat

    model = DeptBedrockChat(model_id=MODEL_ID).bind_tools([get_weather])
    response = model.invoke([HumanMessage("What's the weather in Paris?")])
    print(f"  content: {response.content!r}")
    print(f"  tool_calls: {response.tool_calls}")
    assert response.tool_calls, "adapter did not surface a tool call"
    print("\n  PASS: DeptBedrockChat turned the response into LangChain tool_calls.")


def step_5_tool_result_roundtrip() -> None:
    _hr("5. Multi-turn round-trip (feed a ToolMessage back)")
    from dq_agent.agents.bedrock_chat import DeptBedrockChat

    model = DeptBedrockChat(model_id=MODEL_ID).bind_tools([get_weather])
    first = model.invoke([HumanMessage("What's the weather in Paris?")])
    call = first.tool_calls[0]
    followup = model.invoke([
        HumanMessage("What's the weather in Paris?"),
        AIMessage(content=first.content, tool_calls=first.tool_calls),
        ToolMessage(content=get_weather.invoke(call["args"]), tool_call_id=call["id"]),
    ])
    print(f"  final answer: {followup.content!r}")
    print("\n  PASS: tool_result round-trip completed.")


def _token_counter_model() -> str | None:
    """A model id on this proxy that can count tokens, preferring the configured one.

    Token counting is not enabled for every model: Sonnet 5 returns a 400 from
    /count_tokens naming the ones that do work. Claude models share a tokenizer
    closely enough that counting the same prompt against a sibling is worth acting
    on, so fall back to one rather than give up on the measurement."""
    from dwutils import bedrock

    candidates = [MODEL_ID]
    try:
        candidates += [m for m in bedrock.get_available_models() if m != MODEL_ID]
    except bedrock.BedrockError:
        pass
    for candidate in candidates:
        try:
            info = bedrock.get_model_info(candidate)
        except bedrock.BedrockError:
            continue
        if info.get("capabilities", {}).get("count_tokens"):
            return candidate
    return None


def step_6_prefix_size() -> None:
    """How big is the part of every request that never changes?

    Prompt caching has a minimum below which nothing is stored at all. The tool
    schema and system prompt are the only static part of a scoping request, so if
    they fall short of it the system cache point stores nothing and only the rolling
    message one earns its keep. Counted through the proxy rather than estimated
    locally, because the estimate lands near the line.

    Informational: a failure here says nothing about whether caching works, so this
    never blocks the probe that follows."""
    _hr("6. Static prefix size (bedrock.get_input_token_usage)")
    from dwutils import bedrock

    from dq_agent.agents.bedrock_chat import _to_converse_tools
    from dq_agent.agents.scoping import DEFAULT_RULES_DIR, SYSTEM_PROMPT, _make_tools
    from dq_agent.registry import Registry

    info = bedrock.get_model_info(MODEL_ID)
    caching = info.get("caching", {})
    minimum = caching.get("min_tokens")
    print(f"  cache minimum:  {minimum} tokens")
    print(f"  cache TTLs:     {caching.get('ttls')}")
    print(f"  pricing:        {info.get('pricing')}")

    counter = _token_counter_model()
    if counter is None:
        print("")
        print("  No model on this proxy can count tokens, so the size check is")
        print("  skipped. It only tells us whether the system cache point is worth")
        print("  having; the rolling one is unaffected either way.")
        return
    if counter != MODEL_ID:
        print("")
        print(f"  {MODEL_ID} cannot count tokens; counting against {counter}.")
        print("  Claude models share a tokenizer closely, so treat this as indicative.")

    tools = _to_converse_tools(_make_tools(Registry(DEFAULT_RULES_DIR)))
    request = {
        "model_id": counter,
        "system": [{"text": SYSTEM_PROMPT}],
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        "inferenceConfig": {"maxTokens": 16},
        "toolConfig": {"tools": tools},
    }
    try:
        tokens = bedrock.get_input_token_usage(request)
    except bedrock.BedrockError as error:
        print("")
        print(f"  Could not count tokens ({error.status_code}): {error.message}")
        return

    print("")
    print(f"  tools + system + one message: {tokens} input tokens")
    if minimum and tokens < minimum:
        print("")
        print("  NOTE: below the minimum. The system cache point will store nothing;")
        print("  only the rolling message cache point will, once the transcript grows.")
    else:
        print("")
        print("  PASS: the static prefix is large enough to cache on its own.")


def step_7_prompt_cache() -> bool:
    """Does the proxy accept a cachePoint, and does an identical second call read it?

    `cachePoint` is the one part of the caching wire format dwutils does not
    document, so this is the same go/no-go the toolConfig probe was. Two identical
    calls: the first should report cache *write* tokens, the second should report the
    same prefix as cache *read* tokens."""
    _hr("7. Prompt cache probe (two identical calls with a cachePoint)")
    from dwutils import bedrock

    # Comfortably over any per-model minimum, so a miss means the mechanism is off
    # rather than the prompt being too small to bother storing.
    filler = "You are a careful data quality assistant. " * 400
    request = {
        "model_id": MODEL_ID,
        "system": [{"text": filler}, {"cachePoint": {"type": "default"}}],
        "messages": [{"role": "user", "content": [{"text": "Reply with exactly: one"}]}],
        "inferenceConfig": {"maxTokens": 16},
    }

    def call(label: str) -> dict:
        data = bedrock.invoke(request=request, simple=False, show_usage=False).json()
        usage = data.get("usage", {})
        print(
            f"  {label}: input={usage.get('inputTokens')} "
            f"cache_write={usage.get('cacheWriteInputTokens')} "
            f"cache_read={usage.get('cacheReadInputTokens')}"
        )
        return usage

    try:
        call("first call ")
        second = call("second call")
    except bedrock.BedrockError as error:
        print(f"  FAIL: the proxy rejected the request ({error.status_code}).")
        print(f"  {error.message}")
        if error.status_code == 400:
            print("")
            print("  A 400 means the cachePoint spelling is wrong. Run the app with")
            print("  DQ_AGENT_PROMPT_CACHE=0 until `_with_cache_points` in")
            print("  src/dq_agent/agents/bedrock_chat.py matches what the proxy wants.")
        return False

    print("")
    if second.get("cacheReadInputTokens"):
        print("  PASS: the second call read the first call's cached prefix.")
        return True
    print("  FAIL: no cache read on the second call. The proxy accepted the cachePoint")
    print("  but nothing was reused -- check whether caching is enabled for your")
    print("  account, then keep DQ_AGENT_PROMPT_CACHE=0 so requests stay unmarked.")
    return False


def step_8_cache_ttl() -> None:
    """Will the proxy take a 1-hour cache TTL?

    The default entry lasts 5 minutes. The scoping agent pauses at the approval gate
    for a human to read a contract, which routinely takes longer than that, so the
    entry is often dead exactly when the conversation resumes. The proxy prices both
    `cache_write_5m` and `cache_write_1h`, so a longer TTL is purchasable -- what is
    not documented is how to ask for one.

    This only establishes whether the request is *accepted*; confirming the entry
    really survives an hour would mean waiting one out. Informational either way.
    """
    _hr("8. Cache TTL probe (is a 1-hour cachePoint accepted?)")
    from dwutils import bedrock

    filler = "You are a careful data quality assistant. " * 400
    request = {
        "model_id": MODEL_ID,
        "system": [{"text": filler}, {"cachePoint": {"type": "default", "ttl": "1h"}}],
        "messages": [{"role": "user", "content": [{"text": "Reply with exactly: one"}]}],
        "inferenceConfig": {"maxTokens": 16},
    }
    try:
        data = bedrock.invoke(request=request, simple=False, show_usage=False).json()
    except bedrock.BedrockError as error:
        print(f"  Rejected ({error.status_code}): {error.message}")
        print("")
        print("  So a 1-hour entry cannot be requested this way. Caching still works")
        print("  at the 5-minute default; expect a miss on the turn after a long")
        print("  pause at the approval gate.")
        return

    usage = data.get("usage", {})
    print(
        f"  accepted: cache_write={usage.get('cacheWriteInputTokens')} "
        f"cache_read={usage.get('cacheReadInputTokens')}"
    )
    print("")
    print("  PASS: the proxy accepts a ttl field. Worth using for the rolling cache")
    print("  point so an entry survives the approval gate -- note it is priced at")
    print("  cache_write_1h, which is 2x input rather than 1.25x.")


def main() -> None:
    step_0_env()
    step_1_models()
    step_2_text()
    if not step_3_raw_tool_probe():
        sys.exit(1)
    step_4_adapter_tool_call()
    step_5_tool_result_roundtrip()
    try:
        step_6_prefix_size()
    except Exception as error:  # noqa: BLE001 - informational; step 7 is the go/no-go
        print(f"  Skipped: {error}")
    if not step_7_prompt_cache():
        sys.exit(1)
    try:
        step_8_cache_ttl()
    except Exception as error:  # noqa: BLE001 - informational, like step 6
        print(f"  Skipped: {error}")
    _hr("All checks passed — Bedrock proxy is ready for the scoping agent.")


if __name__ == "__main__":
    main()
