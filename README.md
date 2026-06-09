# dq-agent

A conversational data quality framework for anyone who owns, curates, or delivers a dataset.

Point the tool at a dataset, describe its business context, and it proposes a tested rule
contract through natural language interaction. Once approved, the contract runs
deterministically — no LLM involved at execution time.

---

## How it works

```
dataset + business context
         │
         ▼
  [ Scoping Agent ]      ← converses with you, queries the rule registry
         │
         ▼
  [ Human Approval ]     ← you review, adjust, and sign off the rule suite
         │
         ▼
  [ Execution Engine ]   ← deterministic, runs the approved contract against data
         │
         ▼
  structured results     ← pass/fail per rule, row-level detail, ready for pipelines
```

The LLM is involved only in the scoping conversation and in creative mode (proposing novel
rules not yet in the registry). Profiling and rule execution are pure, testable code.

---

## Project structure

```
dq-agent/
├── src/dq_agent/
│   ├── registry/          # Registry loader — reads rule definitions from YAML
│   ├── engine/            # Execution engine — runs contracts deterministically
│   ├── profiler/          # Dataset profiler + data source connectors
│   │   └── connectors/    # CSV (dev), Postgres (primary target)
│   └── agents/            # LangGraph orchestration (Phase 3+)
├── registry/
│   └── rules/             # Rule definitions as YAML — the core differentiator
├── contracts/
│   └── examples/          # Example approved contracts
├── proposals/             # Creative mode — rule specs awaiting developer review
├── data/
│   └── synthetic/         # Dirty-by-design test dataset
└── tests/
```

---

## Setup

```bash
uv sync
uv sync --extra dev          # include test dependencies
uv sync --extra postgres     # include Postgres connector
uv sync --extra agents       # include LangGraph / LangChain
```

Run tests:

```bash
uv run pytest
```

---

## Development phases

See [ACTION_PLAN.md](ACTION_PLAN.md) for the full roadmap.

| Phase | Focus | Status |
|---|---|---|
| 1 | Rule registry + execution engine | planned |
| 2 | Deterministic profiler (CSV + Postgres) | planned |
| 3 | Scoping agent with human approval gate | planned |
| 4 | Creative mode — novel rule proposals | planned |

---

## Design principles

- **Registry first.** The quality of the rule registry determines the quality of every
  contract the agent produces. Invest there.
- **Deterministic by default.** The LLM is a reasoning layer, not an execution layer.
  Approved rules run as code, not prompts.
- **Framework agnostic at the model layer.** LLM provider is a config value. Swap between
  Anthropic, OpenAI, or a local model without code changes.
- **Own the contract format.** Contracts are portable YAML. Export adapters to other DQ
  tools (Great Expectations, Soda) can be added later without changing the core.
