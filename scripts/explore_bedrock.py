"""Run-at-work probe for the bedrock-proxy integration. NOT production code.

This is a one-time confidence check for the air-gapped work environment. It is
not imported by the agent or engine and never runs in production — it only
verifies that (1) the proxy is reachable, (2) it forwards `tools` and returns
`tool_use` blocks, and (3) DeptBedrockChat round-trips tool calls end to end.
Once Bedrock is confirmed to behave, this script has done its job.

Prereqs (work environment only):
    - `dwutils` importable
    - BEDROCK_PROXY_URL and BEDROCK_TOKEN set (see .env.example)
    - `uv sync --extra agents`

Run:
    uv run python scripts/explore_bedrock.py
"""

from __future__ import annotations

import os
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

MODEL_ID = os.environ.get("DQ_AGENT_MODEL", "eu.anthropic.claude-sonnet-4-6")


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


def step_2_text() -> None:
    _hr("2. Plain text call (bedrock.invoke_simple)")
    from dwutils import bedrock

    result = bedrock.invoke_simple(
        model_id=MODEL_ID, prompt="Reply with exactly: pong", show_usage=False
    )
    print(f"  model_response_string: {result['model_response_string']!r}")


def step_3_raw_tool_probe() -> bool:
    """The go/no-go test: does the proxy forward `tools` and return a tool_use block?"""
    _hr("3. Raw tool-use probe (bedrock.invoke with a tools array)")
    from dwutils import bedrock

    request = {
        "model_id": MODEL_ID,
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": "Use the provided tools to answer.",
        "tools": [{
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }],
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
    }
    data = bedrock.invoke(request=request, show_usage=False).json()
    print(f"  stop_reason: {data.get('stop_reason')}")
    tool_uses = [b for b in data.get("content", []) if b.get("type") == "tool_use"]
    if tool_uses:
        for block in tool_uses:
            print(f"  tool_use -> {block['name']}({block['input']})")
        print("\n  PASS: the proxy forwards tools and returns tool_use blocks.")
        return True
    print("\n  FAIL: no tool_use block. content was:")
    print(f"  {data.get('content')}")
    print("  The proxy may be stripping `tools` — the scoping agent cannot run on it.")
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
