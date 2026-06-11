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
```

## Architecture

The project separates the LLM reasoning layer from all data work. The LLM is intentionally limited to two roles: **scoping conversations** and **creative mode** (proposing novel rules). Everything else is deterministic code.

```text
registry/rules/      ← YAML rule definitions (ID, tags, parameters, execution function)
src/dq_agent/
  registry.py        ← loads and indexes rule YAML at startup
  engine.py          ← runs an approved contract (list of rule IDs + params) against data; no LLM
  rules/             ← rule functions, one module per DQ category
  profiler/          ← produces a structured JSON report (col stats, table stats, semantic hints)
    connectors/      ← adapters that load data into Polars DataFrames (CSV, Postgres)
  agents/            ← LangGraph orchestration (Phase 3+); wraps profiler and registry as tools
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

---

### Build order

Phases are strictly sequential: Phase 1 (registry + engine) and Phase 2 (profiler + connectors) must be complete and tested before any LLM/agent work begins. The registry and engine are independently valuable — the agent is an interface layer on top.
