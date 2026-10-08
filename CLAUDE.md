# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                      # install core deps
uv sync --extra dev          # add pytest / pytest-cov
uv sync --extra postgres     # add ConnectorX for Postgres connector
uv sync --extra agents       # add LangGraph + LangChain

uv run pytest                # run all tests
uv run pytest tests/path/to/test_file.py::test_name   # run a single test
uv run pytest --cov=src/dq_agent                      # run with coverage

uv run python scripts/scoping_cli.py                  # drive the scoping agent from the terminal
uv run python scripts/scoping_cli.py --db scoping.sqlite   # ...with a durable, resumable thread
```

## Architecture

The project separates the LLM reasoning layer from all data work. The LLM is intentionally limited to two roles: **scoping conversations** and **creative mode** (proposing novel rules). Everything else is deterministic code. Execution happens outside this tool: the department runs pipelines only in Airflow (`data-flow` repo) and validates with Great Expectations 0.18.2, so dq-agent's product is a GX-native contract that a shared data-flow task runs.

```text
registry/
  expectations.yaml  ← our notes per allowed expectation: tags, plain-English summary, usage caveats.
                       The rest of the agent's catalogue (descriptions, kwargs) is read from GX
dataflow_kit/        ← staged for the data-flow repo, copied over at work; must not import dq_agent
  dq_validation.py   ← ALLOWED_EXPECTATIONS (names) + policy, load/validate a contract, run it as an
                       in-memory GX suite + checkpoint, raise on failure
  dags/              ← example validation-only Airflow DAG
  dq_contracts/      ← one contract per table: <schema>.<table>.yaml
src/dq_agent/
  registry.py        ← builds the agent's catalogue from GX metadata + registry/expectations.yaml
  models.py          ← Contract: the GX envelope; problems() delegates to the data-flow allowlist
  profiler.py        ← produces a structured JSON report (col stats, table stats, semantic hints);
                       redact() strips raw cell values before any report reaches an LLM
  connectors.py      ← adapters that load data into Polars DataFrames (CSV/Parquet, Postgres)
  agents/            ← LangGraph scoping agent; wraps profiler and registry as tools, human
                       approval gate as an interrupt(), persists approved contract YAML
contracts/           ← approved contracts the agent persists (<schema>.<table>.yaml, git-ignored)
proposals/           ← creative mode outputs — rule specs awaiting developer review, never executed
data/synthetic/      ← dirty-by-design test dataset (committed); all other data excluded by .gitignore
```

### Key design invariants

- **Polars is the profiling representation.** Connectors load into a Polars DataFrame before profiling. Execution is Great Expectations, inside the database, in data-flow.
- **Contracts are GX 0.18.2-native and allowlisted.** Expectation names and kwargs are used directly — no translation layer, no generated code, no `eval`. `dataflow_kit/dq_validation.py` is the single definition of what is valid: `ALLOWED_EXPECTATIONS` lists the permitted names, the permitted kwargs are GX's own keys minus `FORBIDDEN_KWARGS`, values are checked by GX's `validate_configuration` plus a few policy rules. `Contract.problems()` and the agent's catalogue (`registry.py`) both derive from it, so never restate kwargs or types elsewhere. Do not upgrade the GX version silently.
- **Contracts gate execution.** No check runs without an approved contract: data-flow's loader rejects a contract without an `approval` block. Contracts never contain connection strings.
- **Keep the data-flow footprint minimal.** The kit is one utility module, one example DAG and the contracts folder. Do not add scripts or infrastructure there without asking.
- **Proposals are never executed.** `proposals/` contains hypotheses. Promote to the allowlist only after developer review.
- **LLM provider is a config value.** Agent code binds to LangChain's provider-agnostic interface. Swapping between Anthropic, OpenAI, or a local model requires no code changes.

## Roles

Role files in `.claude/roles/` contain task-specific coding criteria. Load the relevant file
before starting work in the corresponding area:

| When you are...                                      | Load this role file              |
| ---------------------------------------------------- | -------------------------------- |
| Editing `registry/expectations.yaml` or the allowlist/policy in `dataflow_kit/dq_validation.py` | `.claude/roles/rule-author.md` |
| Editing the scoping agent's approval gate / interrupt payloads, or the CLI/Streamlit driver that consumes them | `.claude/roles/hitl-driver.md` |

---

### Build order

Phases are strictly sequential: Phase 1 (registry + engine) and Phase 2 (profiler + connectors) must be complete and tested before any LLM/agent work begins. The registry and engine are independently valuable — the agent is an interface layer on top.

---

## Branch: `feat/gx-contracts` (from `feat/bedrock-proxy-adapter`)

This branch replaces the Polars engine with GX-native contracts and the data-flow kit; the engine survives on `feat/bedrock-proxy-adapter`. Like its parent, it targets an **air-gapped work environment** and diverges from `main` on two axes — do not "fix" these back toward `main`:

- **Model.** The default chat model is `DeptBedrockChat` (`src/dq_agent/agents/bedrock_chat.py`), which reaches Anthropic-on-Bedrock through the internal `dwutils.bedrock` proxy. There is no Gemini / `init_chat_model` path and no `GOOGLE_API_KEY`. `dwutils` is internal (not on PyPI) and assumed importable at work; the import is deferred so unit tests run anywhere. `model` stays injectable in `build_graph` for tests.
- **Interface.** There is **no agent-chat-ui and no `langgraph dev` server** — the environment has no npm mirror. The interface is a **local, in-process driver** that compiles the graph itself and supplies its own checkpointer:
  - **CLI:** `scripts/scoping_cli.py` (the supported path today).
  - **Visual (for non-coders):** an optional Streamlit chat panel — same in-process graph + checkpointer pattern, npm-free.

**Proxy behaviour lives in `BEDROCK_PROXY.md`.** The Converse wire format, the `simple=False` calling convention, prompt caching, streaming, per-model cache minimums and pricing, and the error shapes - all verified against the live proxy rather than taken from `dwutils` docstrings, which do not cover most of it. Read it before changing `bedrock_chat.py`, and re-run `scripts/explore_bedrock.py` after any proxy upgrade: these facts have changed under us once already.

**Checkpointer is mandatory for any driver.** The approval gate is a LangGraph `interrupt()`; without a checkpointer it cannot pause and resume, so the contract is never produced. On `main` the `langgraph dev` server supplied this implicitly — here each driver must pass one to `build_graph(checkpointer=...)` (`MemorySaver` for an ephemeral session, `SqliteSaver` for a resumable one). A driver loop must also distinguish a normal conversational turn (send a new message) from a pending interrupt (send `Command(resume=...)`) by checking `result.get("__interrupt__")`.
