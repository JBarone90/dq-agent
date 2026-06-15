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
│   ├── profiler.py        # Dataset profiler — stats, semantic hints, redacted reports
│   ├── connectors.py      # Load CSV/Parquet (dev), Postgres (primary target) into Polars
│   └── agents/            # LangGraph scoping agent + human approval gate (Phase 3)
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

The Postgres integration test (identical profiler report from a file and a live
table) skips unless `DQ_TEST_POSTGRES_URI` is set. To run it
against a throwaway container:

```bash
docker run -d --name dq-test-pg -e POSTGRES_PASSWORD=dq -p 5433:5432 postgres:16
docker exec dq-test-pg psql -U postgres -c "CREATE TABLE orders (
  order_id INTEGER, customer_id TEXT, email TEXT, amount DOUBLE PRECISION,
  status TEXT, created_at DATE, phone TEXT);"
cat data/synthetic/orders.csv | docker exec -i dq-test-pg psql -U postgres \
  -c "\copy orders FROM STDIN WITH (FORMAT csv, HEADER true)"

DQ_TEST_POSTGRES_URI=postgresql://postgres:dq@localhost:5433/postgres uv run pytest
docker rm -f dq-test-pg
```

---

## The two workflows

Everything in this tool belongs to one of two workflows that never overlap in time.
They share only two things: connectors (both start by loading data into Polars) and
the contract YAML (one workflow produces it, the other consumes it).

**Scoping time** — once per dataset, human in the loop, LLM allowed:

```text
connectors.load_csv / load_postgres
      │  Polars DataFrame
      ▼
profiler.profile()  ──►  full report          (raw value examples — stays local)
      │
      │  profiler.redact()
      ▼
redacted report     ──►  scoping conversation (Phase 3: agent + registry tags
      │                                        + your business context)
      ▼
proposed contract   ──►  human approval gate  ──►  contract YAML (persisted)
```

**Run time** — every pipeline run, deterministic, no LLM anywhere:

```text
connectors.load_postgres (fresh data)     approved contract YAML
      │  Polars DataFrame                       │
      └───────────────┬─────────────────────────┘
                      ▼
         engine.run(contract, df, registry)
                      ▼
         list[RuleResult]  ──►  pipeline gates on passed / severity
```

The profiler informs which rules to _propose_; the engine _executes_ what was
approved. The engine never profiles, the profiler never executes, and the LLM
never touches either — it only reads redacted profiler reports during scoping.

### Worked example

The repo ships a dirty-by-design dataset (`data/synthetic/orders.csv` — 20 order
rows with known issues: null customer ids, a duplicate order id, a negative amount,
a malformed email, an unexpected status, a stale date) and an approved contract for
it (`contracts/examples/orders.yaml`).

```python
from pathlib import Path

from dq_agent.connectors import load_csv
from dq_agent.engine import run
from dq_agent.models import Contract
from dq_agent.profiler import profile, redact
from dq_agent.registry import Registry

# scoping time: load, profile, redact
df = load_csv("data/synthetic/orders.csv")
report = profile(df, dataset="orders")
safe = redact(report)          # the only variant an LLM may ever see
```

The redacted report keeps aggregates and hints, drops value examples — the `email`
column profiles as:

```json
{
  "name": "email",
  "dtype": "String",
  "null_rate": 0.0,
  "uniqueness_ratio": 1.0,
  "min": null,
  "max": null,
  "top_values": null,
  "semantic_hint": "email"
}
```

The scoping agent (see below) turns that report plus your business context into a
proposed contract; contracts can also be hand-written. Run time is three lines:

```python
contract = Contract.from_yaml(Path("contracts/examples/orders.yaml"))
results = run(contract, df, Registry(Path("registry/rules")))
```

Every baked-in issue is caught:

```text
min_row_count   PASS  0.0
null_check      FAIL  0.1     # 2 of 20 customer_id values are null
unique_check    FAIL  0.05    # order_id 1001 appears twice
range_check     FAIL  0.05    # one negative amount
allowed_values  FAIL  0.05    # status 'refunded' not in the approved set
regex_match     FAIL  0.05    # 'not-an-email'
freshness       FAIL  0.05    # one order from 2020
```

A pipeline gates on these results directly — e.g. fail the Airflow task when any
rule with severity `error` has `passed == False`. Each result carries its effective
severity (the contract's per-rule override, else the registry default), so the gate
needs nothing but the results.

---

## The scoping agent

Phase 3 wraps the scoping workflow in a single LangGraph agent
(`src/dq_agent/agents/scoping.py`). It converses with the dataset owner, profiles the
dataset (`profile_dataset` — redacted report only, no raw cell values ever reach the
LLM), browses the registry (`list_rules`), and proposes a draft contract
(`propose_contract`, validated against the registry). Approval is a LangGraph
`interrupt()`: the graph pauses, a human accepts / edits / responds, and only an
accepted contract is stamped (`approved_at`, `approved_by`), given a schema snapshot,
and persisted to `contracts/<dataset>.yaml` — directly executable by the engine.

The engine enforces the gate at run time: it raises `ContractNotApprovedError` for
unapproved contracts and `SchemaDriftError` when the live schema no longer matches
the snapshot the contract was scoped against (drift routes the owner back to
re-scoping; it is a contract lifecycle event, not a per-rule failure).

### Running the chat interface

The chat UI is [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui) —
LangChain's off-the-shelf client for LangGraph servers. Three steps:

**1. Configure the model.** The dev default is Gemini 2.5 Flash on Google's free tier:

```bash
uv sync --extra agents
cp .env.example .env       # then put your GOOGLE_API_KEY in .env
```

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
The provider is a config value, not a dependency: set `DQ_AGENT_MODEL` to any
`provider:model` LangChain knows (e.g. `anthropic:claude-sonnet-4-6`, or
`ollama:qwen3:8b` for fully local) and install the matching `langchain-*` package —
no code changes.

**2. Serve the graph.** From the repo root:

```bash
uv run langgraph dev       # serves the 'scoping' graph at http://localhost:2024
```

**3. Connect a chat client.** Quickest is the hosted client — open
[agentchat.vercel.app](https://agentchat.vercel.app) and fill in:

- Deployment URL: `http://localhost:2024`
- Assistant / Graph ID: `scoping`
- LangSmith API key: leave empty (not needed for a local server)

Or run the UI locally instead:

```bash
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui
pnpm install && pnpm dev   # then open http://localhost:3000 and enter the same values
```

Then chat: point the agent at `data/synthetic/orders.csv`, describe the business
context, and iterate on its proposal. When you confirm, the approval gate renders
as an interrupt card (accept / edit / respond); accepting writes the approved
contract to `contracts/<dataset>.yaml`.

> **Free-tier rate limits:** a single scoping turn makes several model requests
> (the agent loop calls the model once per tool round), so Gemini's free-tier
> requests-per-minute cap is easy to hit mid-conversation. If you see 429s, wait
> a minute and continue — the thread keeps its state — or switch `DQ_AGENT_MODEL`
> to a model with a higher free RPM (e.g. `google_genai:gemini-2.5-flash-lite`).

The approval interrupt follows the agent-inbox `HumanInterrupt` schema, so
agent-chat-ui renders the contract review (accept / edit / respond) natively.

---

## Anatomy of a rule

Every rule is two artifacts that share an id: a YAML definition (what the agent and
humans see) and a pure function (what the engine runs).

The YAML in `registry/rules/` carries everything needed to discover, validate, and
route the rule — tags the scoping agent queries, parameter specs that contracts are
validated against, a default severity, and a pointer to the implementation:

```yaml
# registry/rules/null_check.yaml
id: null_check
name: Null Check
description: Fails if the null rate in a column exceeds max_null_rate.
tags: [completeness]
severity: error
parameters:
  column: { type: str, required: true }
  max_null_rate: { type: float, default: 0.0 }
execution:
  module: dq_agent.rules.completeness
  function: null_check
```

The function in `src/dq_agent/rules/` is the implementation: Polars DataFrame in,
`RuleResult` out — no side effects, no I/O, no LLM:

```python
# src/dq_agent/rules/completeness.py
def null_check(df: pl.DataFrame, *, column: str, max_null_rate: float = 0.0) -> RuleResult:
    violation_rate = df[column].null_count() / len(df)
    return RuleResult(
        rule_id="null_check",
        passed=violation_rate <= max_null_rate,
        violation_rate=violation_rate,
    )
```

The registry connects the two at startup: it loads every YAML, indexes rules by id
and tags, and resolves `execution.module` / `execution.function` to the callable on
first use. The engine never imports rule modules directly — all routing goes through
the registry. A contract then activates a rule for one dataset by id, with parameters
chosen during scoping:

```yaml
- rule_id: null_check
  params: { column: customer_id, max_null_rate: 0.0 }
```

For each contract entry the engine validates params against the spec, resolves the
callable, runs it, and folds any failure into that rule's result (`error` set,
`violation_rate` null) — one broken rule never blocks the rest. Empty datasets are
rejected before any rule runs: on zero rows every rule would pass vacuously.

Adding a rule never touches the engine: one YAML file, one function, tests.
Authoring standards live in `.claude/roles/rule-author.md`.

---

## Development phases

See [ACTION_PLAN.md](ACTION_PLAN.md) for the full roadmap.

| Phase | Focus                                   | Status      |
| ----- | --------------------------------------- | ----------- |
| 1     | Rule registry + execution engine        | done        |
| 2     | Deterministic profiler (CSV + Postgres) | done        |
| 3     | Scoping agent with human approval gate  | in progress |
| 4     | Creative mode — novel rule proposals    | planned     |

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
