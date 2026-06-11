# dq-agent

A conversational data quality framework for anyone who owns, curates, or delivers a dataset.

Point the tool at a dataset, describe its business context, and it proposes a tested rule
contract through natural language interaction. Once approved, the contract runs
deterministically — no LLM involved at execution time.

---

## How it works

```text
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

```text
dq-agent/
├── src/dq_agent/
│   ├── registry.py        # Registry loader — reads rule definitions from YAML
│   ├── engine.py          # Execution engine — runs contracts deterministically
│   ├── models.py          # RuleResult, Contract — shared Pydantic models
│   ├── rules/             # Rule functions, one module per DQ category
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

| Phase | Focus                                   | Status  |
| ----- | --------------------------------------- | ------- |
| 1     | Rule registry + execution engine        | done    |
| 2     | Deterministic profiler (CSV + Postgres) | planned |
| 3     | Scoping agent with human approval gate  | planned |
| 4     | Creative mode — novel rule proposals    | planned |

---

## Phase 1 internals: rule → result

This diagram shows how a single rule check flows from configuration to output.

```text
registry/rules/null_check.yaml
  │  id, parameters, execution.module, execution.function
  │
  ▼
Registry (startup)
  │  loads every YAML into RuleDefinition via Pydantic
  │  indexes by rule_id
  │  caches callables via importlib on first resolve()
  │
  ▼
Contract (approved YAML)             DataFrame (Polars)
  │  dataset, list of                │  loaded by a connector
  │  { rule_id, params }             │  (CSV, Postgres, …)
  │                                  │
  └──────────────┬───────────────────┘
                 ▼
           Engine  run()
                 │
                 │  for each ContractRule:
                 │    1. registry.validate_params()   ← required params present?
                 │    2. registry.resolve()           ← importlib → callable
                 │    3. fn(df, **params)             ← pure rule function
                 │    4. catch any error → RuleResult(error=…)
                 │
                 ▼
         list[RuleResult]
           rule_id · passed · violation_rate · error?
```

Key invariants:
- The engine never imports rule modules directly — all routing goes through the registry.
- One failing rule does not block the others; errors are captured per-result.
- No LLM is involved anywhere in this flow. The agent layer (Phase 3) sits above it.

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
