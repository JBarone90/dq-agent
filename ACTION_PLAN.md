# DQ Agent — Action Plan

A conversational data quality framework for anyone who owns, curates, or delivers a dataset.
Through natural language interaction, the tool helps surface appropriate quality rules,
produces an approved contract, and executes it deterministically — without requiring
engineering knowledge to get started.

The LLM earns its place in two places only: **scoping conversations** (interpreting business
context that statistics alone can't capture) and **creative mode** (proposing new rules for
patterns outside the registry). Everything else is deterministic code.

---

## Architecture Overview

```text
User
 │
 ▼
OrchestratorAgent          ← thin, conversation-driven, routes between sub-agents
 ├── ProfilerAgent          ← deterministic core, agent wrapper optional
 └── ContractAgent          ← queries registry, proposes rule suite
 │
 ▼
Human Approval Gate        ← user reviews and approves the rule contract
 │
 ▼
Execution Engine           ← deterministic, no LLM, runs rules against data
 │
 ▼
Result Schema              ← portable output (pass/fail per rule, metadata)
```

Sub-agents have clear contracts and are independently testable. The orchestrator stays thin.
The registry and execution engine are valuable standalone — the agent is an interface layer
on top of them.

---

## Phases

### Phase 1 — Registry + Execution Engine

_Foundation. Build and validate this before any LLM work._

- [ ] Define rule schema: each rule has an ID, name, description, tags (domain, data type,
      severity), parameters, and an execution function
- [ ] Implement a seed registry (10–15 rules): nullability, uniqueness, range checks,
      referential integrity, regex pattern matching, freshness, row count thresholds
- [ ] Build the execution engine: given a dataset and a rule contract (list of rule IDs +
      parameters), run all rules and return a structured result
- [ ] Define the result schema: rule ID, passed/failed, row-level detail, timestamp
- [ ] Write unit tests for every rule and for the engine itself
- [ ] Design registry as config-driven YAML so rules can be added without code changes
      to the engine

**Exit criteria:** given a dataset and a hand-written rule contract, the engine runs all rules
and produces correct, structured results with no LLM involved.

---

### Phase 2 — Deterministic Profiler

_Produce the structured dataset summary the agent will reason over._

- [ ] Column-level stats: null rate, uniqueness ratio, data type, min/max, top-N values,
      value distribution sketch
- [ ] Table-level stats: row count, duplicate rows, schema fingerprint
- [ ] Semantic hints: detect likely email, date, phone, ID columns by pattern
- [ ] Output a structured profiler report (JSON) — this is the agent's primary input
- [ ] Profiler must be backend-agnostic: Polars DataFrames as the internal representation;
      data connectors load into Polars before profiling runs
- [ ] Connectors in scope: local CSV/Parquet (development), Postgres (primary target)

**Exit criteria:** profiler runs on a local file and a Postgres table and produces an identical
structured report format from both sources.

---

### Phase 3 — Scoping Agent (Core LLM Flow)

_Single-agent first, sub-agent split only if it earns its complexity._

- [ ] Set up LangGraph as the orchestration layer with provider-agnostic LLM binding
      (swap model via config, not code changes)
- [ ] `ProfilerAgent`: wraps Phase 2 profiler as a tool callable by the orchestrator
- [ ] `ContractAgent`: takes profiler report + user context, queries registry by tags,
      proposes a parameterized rule suite with reasoning
- [ ] `OrchestratorAgent`: drives the scoping conversation, asks clarifying questions,
      routes between sub-agents, maintains session state
- [ ] Human approval gate: present proposed contract, user can accept, reject individual
      rules, or override parameters before finalizing
- [ ] Persist approved contract as a canonical YAML artifact independent of the agent

**Exit criteria:** user points at a dataset, describes its business context, receives a proposed
rule contract, approves it, and a YAML contract file is produced that the Phase 1 engine can
execute directly.

---

### Phase 4 — Creative Mode

_Novel rule proposals for patterns outside the registry._

- [ ] During scoping, agent identifies dataset characteristics with no matching registry rule
- [ ] Agent drafts a rule specification: name, description, rationale, suggested parameters,
      example pass/fail cases — saved as a proposal file, not an executable rule
- [ ] Proposal format is a lightweight config that a developer can review and promote to
      the registry with minimal effort
- [ ] Track proposals separately from approved rules (different directory, clear status field)
- [ ] Do not execute proposal rules — they are hypotheses until promoted

**Exit criteria:** agent surfaces at least one novel rule proposal during a scoping session on
a dataset with an unusual pattern; proposal file is readable and actionable by a developer.

---

## Test Dataset

A synthetic dataset will be used throughout development to validate each phase in isolation
and end-to-end. It should be defined once and reused across all phases.

Planned characteristics:

- Realistic e-commerce or transactional structure (orders, customers, products)
- Mix of clean and intentionally dirty columns to exercise the rule engine
- Known issues baked in: nulls in non-nullable columns, out-of-range values, duplicates,
  broken referential integrity, stale timestamps, malformed strings (emails, phone numbers)
- Available as both CSV (for local development) and loadable into Postgres (for connector testing)
- Documented feature list so tests can assert specific expected failures

---

## Design Constraints

- **Framework agnostic at the LLM layer**: LLM provider is a config value, not a dependency.
  Swap between OpenAI, Anthropic, local models without code changes.
- **Deployment agnostic**: no assumptions about cloud provider. Runs locally; designed to
  containerize for internal deployment.
- **Monitoring is out of scope for this tool.** The execution engine produces a result
  schema; downstream pipelines own scheduling, storage, and alerting.
- **No rule is ever executed without human approval of the contract.**
- **The registry is the differentiator.** Invest in rule quality, tagging, and parameterization.
  A weak registry makes the agent unreliable regardless of prompt quality.
- **Own the contract format.** The canonical rule contract is a clean, expressive YAML
  defined by this tool. Export adapters to Great Expectations or Soda can be added later
  as optional plugins if there is real demand — their schemas should not constrain phase 1
  rule design.

---

## Tech Stack (Initial)

| Layer                | Choice             | Rationale                                                                            |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------ |
| Language             | Python             | Ecosystem fit for data tooling                                                       |
| Package manager      | uv                 | Fast, reproducible                                                                   |
| Orchestration        | LangGraph          | Native support for multi-agent + human-in-the-loop                                   |
| LLM binding          | LangChain core     | Provider-agnostic interface                                                          |
| Data layer           | Polars             | Fast, memory-efficient, clean API; consistent interface across CSV and DB connectors |
| DB connector         | ConnectorX or ADBC | Polars-native Postgres ingestion                                                     |
| Rule/contract format | YAML               | Human-readable, diffable, versionable                                                |
| Testing              | pytest             | Standard                                                                             |

---

## Open Questions

- Which Postgres connector integrates most cleanly with Polars in practice? (ConnectorX vs ADBC)
- Should the CLI be the only interface for v1, or is a minimal web UI worth scoping early?
- Multi-tenancy: single user for now; if internal deployment, does the registry need
  access control per team or dataset?
