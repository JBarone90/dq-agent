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

The project separates the LLM reasoning layer from all data work. The LLM is intentionally limited to two roles: **scoping conversations** and **creative mode** (proposing novel rules). Everything else is deterministic code.

```text
registry/rules/      ← YAML rule definitions (ID, tags, parameters, execution function)
src/dq_agent/
  registry.py        ← loads and indexes rule YAML at startup
  engine.py          ← runs an approved contract (list of rule IDs + params) against data; no LLM
  rules/             ← rule functions, one module per DQ category
  profiler.py        ← produces a structured JSON report (col stats, table stats, semantic hints);
                       redact() strips raw cell values before any report reaches an LLM
  connectors.py      ← adapters that load data into Polars DataFrames (CSV/Parquet, Postgres)
  agents/            ← LangGraph scoping agent; wraps profiler and registry as tools, human
                       approval gate as an interrupt(), persists approved contract YAML
contracts/examples/  ← approved contract YAML artifacts; output of the human approval gate
proposals/           ← creative mode outputs — rule specs awaiting developer review, never executed
data/synthetic/      ← dirty-by-design test dataset (committed); all other data excluded by .gitignore
```

### Key design invariants

- **Polars is the internal data representation.** Connectors load into a Polars DataFrame before any profiling or rule execution runs. Never pass raw DB cursors or pandas frames into the engine.
- **Rules are config-driven YAML.** Adding a rule must not require changes to the engine. The engine calls the function referenced in the YAML; rules are the config, the engine is the runner.
- **Contracts gate execution.** No rule runs without an approved, persisted YAML contract. The contract format is canonical and owned by this tool — not derived from Great Expectations or Soda schemas.
- **Proposals are never executed.** `proposals/` contains hypotheses. Promote to `registry/rules/` only after developer review.
- **LLM provider is a config value.** Agent code binds to LangChain's provider-agnostic interface. Swapping between Anthropic, OpenAI, or a local model requires no code changes.

## Roles

Role files in `.claude/roles/` contain task-specific coding criteria. Load the relevant file
before starting work in the corresponding area:

| When you are...                                      | Load this role file              |
| ---------------------------------------------------- | -------------------------------- |
| Writing or editing files in `registry/rules/` or `src/dq_agent/rules/` | `.claude/roles/rule-author.md` |
| Editing the scoping agent's approval gate / interrupt payloads, or the CLI/Streamlit driver that consumes them | `.claude/roles/hitl-driver.md` |

---

### Build order

Phases are strictly sequential: Phase 1 (registry + engine) and Phase 2 (profiler + connectors) must be complete and tested before any LLM/agent work begins. The registry and engine are independently valuable — the agent is an interface layer on top.

---

## Branch: `feat/bedrock-proxy-adapter`

This branch targets an **air-gapped work environment** and diverges from `main` on two axes — do not "fix" these back toward `main`:

- **Model.** The default chat model is `DeptBedrockChat` (`src/dq_agent/agents/bedrock_chat.py`), which reaches Anthropic-on-Bedrock through the internal `dwutils.bedrock` proxy. There is no Gemini / `init_chat_model` path and no `GOOGLE_API_KEY`. `dwutils` is internal (not on PyPI) and assumed importable at work; the import is deferred so unit tests run anywhere. `model` stays injectable in `build_graph` for tests.
- **Interface.** There is **no agent-chat-ui and no `langgraph dev` server** — the environment has no npm mirror. The interface is a **local, in-process driver** that compiles the graph itself and supplies its own checkpointer:
  - **CLI:** `scripts/scoping_cli.py` (the supported path today).
  - **Visual (for non-coders):** an optional Streamlit chat panel — same in-process graph + checkpointer pattern, npm-free.

**Checkpointer is mandatory for any driver.** The approval gate is a LangGraph `interrupt()`; without a checkpointer it cannot pause and resume, so the contract is never produced. On `main` the `langgraph dev` server supplied this implicitly — here each driver must pass one to `build_graph(checkpointer=...)` (`MemorySaver` for an ephemeral session, `SqliteSaver` for a resumable one). A driver loop must also distinguish a normal conversational turn (send a new message) from a pending interrupt (send `Command(resume=...)`) by checking `result.get("__interrupt__")`.
