"""The scoping agent's default chat model: Claude through the department's bedrock-proxy.

The proxy serves the Anthropic Messages API at `ANTHROPIC_BASE_URL`, authenticated with
`ANTHROPIC_AUTH_TOKEN` (both set in the work tool environments), so LangChain's own
`ChatAnthropic` talks to it directly. Tool calls, streaming, token usage and retries
come from `langchain-anthropic` and the `anthropic` SDK; nothing here translates a
wire format.
"""

from __future__ import annotations

import os

import anthropic
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

# As listed by the proxy's /anthropic/v1/models. The bare `claude-sonnet-5-5` is also
# accepted, but only the listed `eu.` profile is known to keep inference in the EU.
DEFAULT_MODEL_ID = "eu.anthropic.claude-sonnet-5-5"
MAX_TOKENS = 16000  # thinking counts towards it; higher needs streaming for every call

# 1h rather than 5m: the agent waits at the approval gate while a human reads the
# contract, which routinely outlasts five minutes, and an expired entry re-writes the
# whole transcript.
CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}

ModelError = anthropic.APIError
"""What a driver catches for a failed model call: every SDK error, status and connection
alike, including the subclasses `langchain-anthropic` re-raises them as."""

_ENV = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN")


def default_model(model_id: str = DEFAULT_MODEL_ID, *, cache_prompt: bool = True) -> ChatAnthropic:
    """`ChatAnthropic` pointed at the proxy.

    Caching is the API's automatic kind: a top-level `cache_control` marks the end of
    each request, which covers tools, system prompt and transcript. It rolls forward on
    its own, since each turn resends the previous request plus a few messages.

    The credentials are passed explicitly: `ChatAnthropic` does not read
    `ANTHROPIC_AUTH_TOKEN`, and built from the environment alone it failed against the
    proxy (BEDROCK_PROXY.md). Without any base URL it would aim at api.anthropic.com,
    which the work network cannot reach.
    """
    missing = [name for name in _ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not set: the bedrock-proxy is only available in the "
            "work environment. To run elsewhere, inject a model: build_graph(model=...)."
        )
    # mypy reads the fields' aliases (model_name, max_tokens_to_sample) as the only names
    return ChatAnthropic(  # type: ignore[call-arg]
        model=model_id,
        base_url=os.environ["ANTHROPIC_BASE_URL"],
        api_key=SecretStr(os.environ["ANTHROPIC_AUTH_TOKEN"]),
        max_tokens=MAX_TOKENS,
        model_kwargs={"cache_control": CACHE_CONTROL} if cache_prompt else {},
    )
