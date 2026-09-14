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


def main() -> None:
    step_0_env()
    step_1_models()
    step_2_text()
    if not step_3_raw_tool_probe():
        sys.exit(1)
    step_4_adapter_tool_call()
    step_5_tool_result_roundtrip()
    _hr("All checks passed — Bedrock proxy is ready for the scoping agent.")


if __name__ == "__main__":
    main()
