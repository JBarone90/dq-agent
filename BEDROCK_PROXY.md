# Bedrock proxy reference

How the internal `dwutils.bedrock` proxy actually behaves, including the parts its docstrings do not cover. Everything here was verified against the live proxy on 2026-09-14 with a probe script (`scripts/explore_bedrock.py`, since removed; it is in git history). Re-check these facts after any proxy upgrade, because several of them changed once already and broke the pipeline silently.

## Calling convention

`invoke()` has two modes and the default is not the one this project uses.

Simple mode (the default) takes `model_id=` and `prompt=` and returns a dict. Full request mode takes a Converse `request=` dict and returns the raw `Response`, and it **must** be selected with `simple=False`. Passing `request=` without it raises a `ValueError` before any network call is made, so the failure looks nothing like a proxy problem.

```python
bedrock.invoke(request=body, simple=False, show_usage=False, max_retries=3)
```

## Wire format: Converse, not Anthropic Messages

The proxy speaks the Bedrock Converse shape. A request mixes the proxy's own snake_case `model_id` with camelCase Converse fields, so it is not a pure passthrough and the naming is genuinely inconsistent.

| Anthropic Messages | Converse |
| --- | --- |
| `"system": "text"` | `"system": [{"text": "..."}]` |
| `"max_tokens": N` | `"inferenceConfig": {"maxTokens": N}` |
| `"stop_sequences": [...]` | `"inferenceConfig": {"stopSequences": [...]}` |
| `"tools": [...]` | `"toolConfig": {"tools": [{"toolSpec": ...}]}` |
| `"anthropic_version"` | dropped |
| `{"type": "text", "text": "..."}` | `{"text": "..."}` |
| `{"type": "tool_use", "id", "name", "input"}` | `{"toolUse": {"toolUseId", "name", "input"}}` |
| `{"type": "tool_result", "tool_use_id", "content": str}` | `{"toolResult": {"toolUseId", "content": [{"text": ...}]}}` |
| reply at `content` | reply at `output.message.content` |
| `stop_reason` | `stopReason` |
| `usage.input_tokens` | `usage.inputTokens` |

No content block carries a `"type"` key: Converse identifies a block by which key it holds. This matters for the tool-result coalescing in `_to_converse` - every user turn is now a list of blocks, so "is the content a list" cannot stand in for "is this an open tool-result turn".

## Undocumented, but confirmed working

`dwutils.bedrock` documents `system` / `messages` / `inferenceConfig` and nothing else. Two shapes the agent depends on were probed rather than read from docs:

- **`toolConfig` with `toolSpec` / `inputSchema.json`** - accepted, and the model returns `toolUse` blocks. Step 3 of the probe.
- **`cachePoint`, including a `ttl` field** - accepted and honoured. Steps 7 and 8.

If either is ever rejected with a 400, the request shape is the first suspect, not the proxy.

## Per-model facts

Measured 2026-09-14. **These vary by model - do not assume one model's numbers apply to another.**

| | Sonnet 5 | Sonnet 4.6 |
| --- | --- | --- |
| cache minimum | 4096 tokens | 1024 tokens |
| cache TTLs | 5m, 1h | 5m, 1h |
| input / output per M | $2.20 / $11.00 | $3.30 / $16.50 |
| cache read per M | $0.22 | $0.33 |
| cache write 5m / 1h per M | $2.75 / $4.40 | $4.125 / $6.60 |
| `/count_tokens` | **not available** | available |

The cache minimum quadrupling between 4.6 and 5 is what killed an earlier design that cached the tool schema and system prompt separately: that prefix measures 1862 tokens, so on Sonnet 5 it can never be stored on its own.

`get_model_info` works for every model, including ones `/count_tokens` rejects. Check `capabilities.count_tokens` before counting, and fall back to a sibling model - Claude models share a tokenizer closely enough for the number to be actionable.

## Prompt caching as used here

One cache point, on the last message, asking for a 1-hour TTL. See `_with_cache_points` in `src/dq_agent/agents/bedrock_chat.py`.

Converse assembles a request as tools, then system, then messages, so a marker on the final message covers everything. It rolls forward by itself: each turn resends the previous turn's request plus a few new messages, so the stored prefix is matched automatically and a slightly longer one is stored. **Reads need no marker** - Bedrock matches the longest stored prefix it holds and bills those tokens at the cache rate. A marker only says how much to store.

The 1-hour TTL is bought deliberately. The approval gate pauses for a human to read a contract, routinely past five minutes, and an expired entry means paying to write the whole transcript again rather than extending it. The higher write rate applies only to the tokens each turn adds; an expiry rewrites everything.

Economics: a token pays for itself once it has been re-sent three times. Nothing is stored until a request clears the model's minimum, so short sessions cost exactly what they always did. On a modelled 8-turn session over a 50-column table, input cost drops about 55%; output tokens are unaffected.

`DQ_AGENT_PROMPT_CACHE=0` disables it if the proxy ever stops accepting `cachePoint`.

## Streaming

`invoke(..., stream=True)` returns an iterator of raw Converse events. Two consequences worth knowing:

**No usage headers at all.** HTTP headers are sent before the first token exists, so the proxy cannot stamp `x-cost` or `x-tokens-used` on a stream. Token counts arrive in the closing `metadata` event instead. The adapter does not track cost: it reports token counts only (`usage_metadata`, cache reads and writes included), streamed or not.

**Failures can start after the iterator is handed back.** The proxy reports a mid-stream failure as an error event, which dwutils raises while iterating, long after the status code is gone. `_stream` translates those too, so a driver's `except BedrockProxyError` covers streamed turns.

## Errors

`BedrockError` subclasses `RuntimeError` and carries `.status_code`, so `except RuntimeError` still catches it and the cause can be branched on without parsing text.

- **400** - the request body was rejected. On this project that usually means a Converse shape problem.
- **403** - either the token is unauthorised, or the daily budget is spent. The budget one clears at midnight UTC, and `get_usage()` says which it is.
- **429 / 5xx** - retried automatically with backoff, honouring `Retry-After`. What surfaces has already exhausted `max_retries`.

Missing `BEDROCK_PROXY_URL` / `BEDROCK_TOKEN` raises a plain `ValueError`, not a `BedrockError`. `_invoke` catches both, because the drivers only know about `BedrockProxyError`.

## Environment

`BEDROCK_PROXY_URL` and `BEDROCK_TOKEN` are read by `dwutils.bedrock` itself, not by this project, and are admin-provisioned in the work environment. `DQ_AGENT_MODEL` and `DQ_AGENT_PROMPT_CACHE` are ours. Nothing on this branch loads a `.env` file automatically - see `.env.example`.

## Unrelated noise

Proxy calls in the work environment print `ignoring unparsable config .../pyproject.toml`, twice. Our `pyproject.toml` parses cleanly under `tomllib` on 3.11+ (the project now needs 3.13), so this is some other tool in that environment reading it with an older or stricter parser. Harmless, but it is not coming from this project.

## Re-verifying

The removed probe script (`git show 783cf0d:scripts/explore_bedrock.py`) - steps 0-2 check reachability and a plain call, 3 proves tool calling still works, 4-5 drive the adapter end to end, 6 reports the static prefix against the cache minimum, 7 proves caching round-trips, 8 proves the TTL is honoured by reading it off the bill rather than trusting a 200.

Steps 3 and 7 are go/no-go and exit non-zero. Steps 6 and 8 are informational and never block the run.
