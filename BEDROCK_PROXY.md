# Bedrock proxy reference

How the department's bedrock-proxy behaves for this project. Unless marked otherwise, everything here was verified against the live proxy on 2026-11-03 with a probe run at work. The proxy is a pilot and has broken its clients before, so re-check these facts after any proxy upgrade: `uv run pytest -m integration -v` (`tests/test_live_proxy.py`) covers what the agent depends on.

## What changed (2026-11)

`dwutils.bedrock` was removed. Importing it now raises `BedrockRemovedError`, and the maintainers ask that it not be pinned back or recreated. Its Converse-style `/invoke` endpoint is deprecated in favour of an Anthropic Messages API endpoint, so the official clients work directly. The old adapter (`DeptBedrockChat`, which translated LangChain messages to Converse) is gone. This file previously documented the Converse wire format and `dwutils` calling convention; that version is in git history (`git show feat/gx-contracts:BEDROCK_PROXY.md`).

## Endpoint and credentials

`ANTHROPIC_BASE_URL` (`http://<proxy>:5000/anthropic`) and `ANTHROPIC_AUTH_TOKEN` are set in the work tool environments. The SDK appends `/v1/messages`, `/v1/models` and so on.

- **The `anthropic` SDK** picks both variables up by itself: `anthropic.Anthropic()` works.
- **`ChatAnthropic` does not.** Built from the environment alone, it failed with `AnthropicConnectionError`. Its source (1.7.9) reads the base URL from `ANTHROPIC_API_URL`, then `ANTHROPIC_BASE_URL`, but the key only from `ANTHROPIC_API_KEY`, never `ANTHROPIC_AUTH_TOKEN`. Why that surfaced as a connection error rather than an authentication error was not pinned down. `src/dq_agent/agents/llm.py` therefore passes `base_url` and `api_key` explicitly. The token works both as `api_key` (sent as `x-api-key`) and as an `Authorization: Bearer` header.

`BEDROCK_PROXY_URL` / `BEDROCK_TOKEN` (the old variables) are still set but nothing here uses them.

## Models

`GET /anthropic/v1/models` lists Bedrock-style ids with an `eu.` prefix, e.g. `eu.anthropic.claude-sonnet-5-5`, `eu.anthropic.claude-haiku-5-5`, `eu.anthropic.claude-opus-5-5`, plus older generations. The bare names (`claude-sonnet-5-5`) are accepted too, but that aliasing is undocumented and nothing confirms it routes to the same EU inference profile, so the default uses the listed id. An unknown id is a 404 (below).

The default is Sonnet 5.5. `DQ_AGENT_MODEL` overrides it without a code change, e.g. `eu.anthropic.claude-haiku-5-5` to try the cheaper model.

## Verified working through `ChatAnthropic`

- A plain call, with usage on `usage_metadata`.
- Tool calling: `bind_tools` returns `tool_calls`, and a tool-result round trip completes.
- Streaming, text and tool calls.
- Prompt caching with a block-level `cache_control` on the system prompt: `input_token_details.cache_read` was reported (6003 of 6012 input tokens).

**Not yet verified:** the agent uses the API's *automatic* caching, a top-level `cache_control` on the request, because it covers the whole transcript and rolls forward each turn without our code marking messages. The probe tested only the block-level form. `test_a_repeated_prefix_is_read_from_the_cache` and the session test in `tests/test_live_proxy.py` check it. If they fail, or the proxy rejects the field, run with `DQ_AGENT_PROMPT_CACHE=0` until it is fixed.

## Prompt caching as used here

The TTL is 1 hour, bought deliberately. The approval gate pauses for a human to read a contract, routinely past five minutes, and an expired entry means paying to write the whole transcript again rather than extending it. The higher write rate applies only to the tokens each turn adds.

Each model has a minimum cacheable prefix, and shorter prefixes silently are not cached. On the old proxy Sonnet 5's minimum was 4096 tokens (measured 2026-09-14). The agent's tools plus system prompt measured 1862 tokens then, so early turns may not cache. Later turns, which carry profile reports, clear any minimum.

## Errors

The SDK raises its typed errors; `langchain-anthropic` re-raises them as subclasses (`AnthropicModelNotFoundError` is still an `anthropic.NotFoundError`). The drivers catch `anthropic.APIError`, exported as `dq_agent.agents.llm.ModelError`, which covers status errors and connection errors alike.

- **404** on an unknown model: `{'type': 'error', 'error': {'type': 'not_found_error', 'message': '[E3] Model no-such-model is not available. Use GET /anthropic/v1/models to see available models.'}}`.
- **Connection error** when the base URL is wrong or unreachable.
- **429 / 5xx** and connection errors are retried by the SDK (2 retries, with backoff) before they surface.

Not yet known: whether the daily budget the old proxy enforced still exists and how hitting it looks (it was a 403 that cleared at midnight UTC), and whether there is a replacement for `dwutils.bedrock.get_usage()`.

## Unrelated noise

Proxy calls in the work environment used to print `ignoring unparsable config .../pyproject.toml`, twice. Our `pyproject.toml` parses cleanly under `tomllib`, so this was some other tool in that environment reading it with an older or stricter parser. Harmless, and not coming from this project.
