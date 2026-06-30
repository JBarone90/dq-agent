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

- [x] Define rule schema: each rule has an ID, name, description, tags (domain, data type,
      severity), parameters, and an execution function
- [x] Implement a seed registry (10–15 rules): nullability, uniqueness, range checks,
      regex pattern matching, freshness, row count thresholds
      (referential integrity deferred — single-table scope, see Design Decisions)
- [x] Build the execution engine: given a dataset and a rule contract (list of rule IDs +
      parameters), run all rules and return a structured result
- [x] Define the result schema: rule ID, passed/failed, row-level detail, timestamp
- [x] Write unit tests for every rule and for the engine itself
- [x] Design registry as config-driven YAML so rules can be added without code changes
      to the engine

**Exit criteria:** given a dataset and a hand-written rule contract, the engine runs all rules
and produces correct, structured results with no LLM involved.

---

### Phase 2 — Deterministic Profiler

_Produce the structured dataset summary the agent will reason over._

- [x] Column-level stats: null rate, uniqueness ratio, data type, min/max, top-N values,
      value distribution sketch
- [x] Table-level stats: row count, duplicate rows, schema fingerprint
- [x] Semantic hints: detect likely email, date, phone, ID columns by pattern
- [x] Output a structured profiler report (JSON) — this is the agent's primary input
- [x] Profiler must be backend-agnostic: Polars DataFrames as the internal representation;
      data connectors load into Polars before profiling runs
- [x] Connectors in scope: local CSV/Parquet (development), Postgres (primary target)
- [x] Redacted report variant for LLM consumption: aggregates, types, null rates, and
      semantic hints only — no raw cell values (top-N examples excluded by default).
      Required even with an approved cloud tenant; see Design Decisions

**Exit criteria:** profiler runs on a local file and a Postgres table and produces an identical
structured report format from both sources.

---

### Phase 3 — Scoping Agent (Core LLM Flow)

_Single-agent first, sub-agent split only if it earns its complexity._

- [x] Set up LangGraph as the orchestration layer with provider-agnostic LLM binding
      (swap model via config, not code changes) — `init_chat_model` driven by
      `DQ_AGENT_MODEL`; graph in `src/dq_agent/agents/scoping.py`
- [x] `ProfilerAgent`: wraps Phase 2 profiler as a tool callable by the orchestrator
      — implemented as the `profile_dataset` tool (single-agent, per the phase intro)
- [x] `ContractAgent`: takes profiler report + user context, queries registry by tags,
      proposes a parameterized rule suite with reasoning — implemented as the
      `list_rules` + `propose_contract` tools, validated against the registry
- [x] `OrchestratorAgent`: drives the scoping conversation, asks clarifying questions,
      routes between sub-agents, maintains session state — the single scoping agent
      covers this; promote to sub-agents only if complexity demands it
- [x] Human approval gate: present proposed contract, user can accept, reject individual
      rules, or override parameters before finalizing — LangGraph `interrupt()` in the
      `approval` node, project HITL schema (approve / edit / reject)
- [x] Persist approved contract as a canonical YAML artifact independent of the agent
      — approval stamps `approved_at`/`approved_by` and writes `contracts/<dataset>.yaml`
- [x] Enforce approval in the engine (deferred from Phase 1): `run()` refuses contracts
      without `approved_at` set — raises `ContractNotApprovedError`
- [x] Severity model (design agreed, deferred from Phase 1): registry YAML provides the
      default severity per rule; the contract can override it per dataset (severity is a
      property of the rule *in context*, not of the rule itself); the engine stamps the
      effective severity into each `RuleResult` so downstream consumers can act on a
      result alone — fail on `error`, log on `warning` — without re-joining the registry
- [x] Evaluation harness for the scoping agent: hand-write an "ideal contract" for each
      synthetic dataset with documented issues, then score the agent's proposal against
      it (issues caught / missed / spurious rules). This is the regression suite for
      prompt, model, and registry changes — human approval validates one session,
      the harness validates the agent
- [x] Record approver identity: `approved_by` alongside `approved_at`; design the
      contract format so a fuller audit trail can be added later without breaking it
- [x] Schema drift invalidates the contract: pre-flight check compares contract columns
      against the live schema; on drift, route the owner back to re-scoping instead of
      failing rule-by-rule — the contract snapshots `columns` (name → dtype) at
      proposal time; `run()` raises `SchemaDriftError` naming what drifted
- [x] Human-readable run report: deterministic renderer from `list[RuleResult]` to a
      summary a non-technical dataset owner can read
- [x] Wire `load_postgres_profiling` into the agent as a `profile_table` tool so the agent
      can scope a Postgres table, not just a local file (kept separate from the file-only
      `profile_dataset` rather than overloading it). The tool accepts a table name and
      resolves the connection URI from the environment via `connectors.resolve_dsn`
      (`DATABASE_DSN__datasets_1`, mirroring how `build_graph` resolves `DQ_AGENT_MODEL`) —
      never from the LLM, since a URI carries credentials (government-data privacy). Loading
      is adaptive against the deterministic `PROFILE_MAX_ROWS` cap: the planner estimate
      decides full load vs. `TABLESAMPLE`, and the tool passes `sampled=` straight through to
      `profiler.profile` so a sampled report is flagged honestly. The model only supplies the
      locator. Files keep their current full-load path (local, no transfer cost).
      Sampled-profile guardrails: the prompt steers bounds toward owner-stated domain limits
      (not observed extremes), and `propose_contract` confirms a proposed `range_check`
      against the live table with an exact `connectors.column_bounds` query — but only when
      the profile is sampled *and* a range_check references a column, so no wasted compute.
- [x] Terminal driver for the scoping conversation: `scripts/scoping_cli.py` runs the graph
      in-process, supplies its own checkpointer, and handles the full converse + approval loop
      (distinguishing a normal turn from a pending `interrupt()` by checking `__interrupt__`).
      This is the air-gapped replacement for a chat server — see the 2026-06-29 design entry
- [~] Streamlit chat panel for non-technical owners (`app/scoping_app.py`, `ui` extra):
      same in-process graph + checkpointer pattern as the CLI, npm-free (Streamlit's frontend
      ships in its wheel). **Scaffolded:** chat transcript, tool-call visibility with a toggle,
      and an approve/edit/reject gate (edit exposes the contract YAML) all wired to the real
      graph. **Pending an adapter change:** token-usage readout (needs `DeptBedrockChat` to
      surface `usage_metadata`) and token streaming (needs a `_stream` on the adapter +
      streaming support in the proxy). Auth, deployment, and multi-user are explicitly
      deferred (decide with IT later)

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

### Deferred — Registry-driven contract summaries

`report.describe_contract()` renders a contract as plain English for the approval gate, but it does so with a per-`rule_id` `if/elif` chain. That couples the renderer to the rule catalogue: adding a rule to `registry/rules/` requires editing `report.py` to phrase it well, which cuts against the registry invariant ("adding a rule is just YAML + a function, never a core-code change"). It degrades gracefully today (unknown rules fall back to the registry name), so this is a scalability cleanup, not a bug.

- [ ] Move phrasing into the rule YAML as an optional `summary` template field (e.g. `` "`{column}` must never be empty" ``); `describe_contract` renders it generically via `str.format(**params)`, falling back to rule name + params when absent. New rules then ship their own description with no `report.py` edit.
- [ ] Decide how to handle the few rules whose phrasing is conditional on a value (`null_check` at `max_null_rate: 0` vs. > 0; `range_check` one bound vs. two) — either accept blunter unconditional wording or keep a tiny override map only for those.
- [ ] Optional, creative-mode-adjacent: a small agent that *drafts* the `summary` template when a developer adds a rule, for human review before it is frozen as config. LLM at authoring time only.

Explicitly out of scope: an LLM that translates the contract *at the approval gate* (runtime). The approval card is the surface a human reads to decide whether to approve, so it must faithfully represent the contract — a generated summary could drop a rule or misstate a threshold, and it adds nondeterminism, latency, and another redaction surface to a safety-critical gate. Keep the LLM at authoring time, never in the render/approval path.

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

## Design Decisions (interview, 2026-06-11)

Explicit answers to questions the original plan left open.

**Purpose & audience.** Dual ambition: an internal tool at work (government context) and a
portfolio/learning project. Reliability and data privacy are first-class concerns, not
nice-to-haves. The product's primary deliverable is the **approved contract**, not the run:
users scope and approve a contract through the tool; scheduled pipelines execute it later.

**Privacy (government data).** LLM calls go only through an approved cloud tenant (e.g. a
compliant Azure OpenAI deployment). As defense in depth, profiler reports sent to the LLM
must still contain no raw cell values — aggregates, types, null rates, and semantic hints
only. This is a hard requirement on the Phase 2 report format, not a Phase 3 afterthought.

**Scope.** Single-table contracts for v1. Referential integrity (multi-table) is deferred,
and the engine keeps its `run(contract, df, registry)` shape until a concrete multi-table
need arrives. Dataset scale is unknown and varies; work tables live in Postgres. Design the
profiler for medium scale with sampling hooks; revisit SQL pushdown only if real tables
demand it.

**Execution & integration.** dq-agent is consumed as a **library** by pipelines: an Airflow
task imports a connector + the engine, loads the approved contract, and gates on the
results. Rationale: the engine is already pure functions over in-memory data with
structured return values — a library call gives the pipeline typed results with no
subprocess management or output parsing. A CLI can be added later as a thin wrapper over
the same entry points if operational needs call for it (e.g. running in an image without
installing dq-agent into the DAG environment).

**Results.** Consumers need a human-readable report (summary for non-technical dataset
owners) and machine-readable results for pipeline gating. Run history, trends, and alerting
remain out of scope — downstream pipelines own them.

**Trust & governance.** The registry is curated by a small engineering team via PR review;
the rule-author role doc is the team-facing standard. Contract approval records the
approver's identity (`approved_by`), not just the timestamp. A schema change after approval
invalidates the contract and routes the owner back to re-scoping — drift is a contract
lifecycle event, not a per-rule runtime error.

**Interface.** v1 is a minimal web UI (chat panel + contract review screen), built as a
localhost demo first. Authentication, deployment, and multi-user support are deliberately
deferred until the demo proves the workflow and IT can be involved.

**Interface implementation (2026-06-12).** _(Superseded on `feat/bedrock-proxy-adapter` — see
the 2026-06-29 entry below.)_ The demo UI is agent-chat-ui rather than a custom build.
Consequences for Phase 3 design: the scoping agent must be exposed as a LangGraph Server graph
(`langgraph.json` + `langgraph dev`), and the human approval gate must be implemented as a
LangGraph `interrupt()` whose payload follows the off-the-shelf HITL schema (`action_requests`
+ `review_configs`), so the UI can render approve/edit/reject controls without custom front-end
work. If the demo later needs a bespoke contract-review screen, that is an additive replacement
— the graph and interrupt contract stay the same.

**Interface implementation, branch revision (2026-06-29, `feat/bedrock-proxy-adapter`).** The
work environment is air-gapped with **no npm mirror**, so agent-chat-ui (and any Node-based UI)
is not installable, and the `langgraph dev` server it depends on is dropped. The interface is a
**local, in-process driver** instead: a terminal CLI (`scripts/scoping_cli.py`) today, with an
optional **Streamlit** chat panel for non-technical owners (Streamlit ships its frontend inside
the Python wheel — no Node). The graph and the `interrupt()` HITL contract are unchanged — only
the consumer changed — so this is exactly the "additive replacement" the 2026-06-12 entry
anticipated. **Load-bearing consequence:** the `langgraph dev` server used to provide the
checkpointer implicitly; an in-process driver must compile the graph with its own
(`build_graph(checkpointer=...)`) or the `interrupt()` approval gate cannot pause/resume and no
contract is produced. `MemorySaver` for an ephemeral run, `SqliteSaver` (`langgraph-checkpoint-
sqlite`) for a resumable one.

---

## Tech Stack (Initial)

| Layer                | Choice             | Rationale                                                                            |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------ |
| Language             | Python             | Ecosystem fit for data tooling                                                       |
| Package manager      | uv                 | Fast, reproducible                                                                   |
| Orchestration        | LangGraph          | Native support for multi-agent + human-in-the-loop                                   |
| LLM binding          | LangChain core     | Provider-agnostic interface                                                          |
| Dev/demo LLM         | Bedrock proxy (`DeptBedrockChat`) | Air-gapped default on this branch: Anthropic-on-Bedrock via the internal `dwutils.bedrock` proxy; `DQ_AGENT_MODEL` picks the Bedrock model id (`main` uses Gemini free tier) |
| Data layer           | Polars             | Fast, memory-efficient, clean API; consistent interface across CSV and DB connectors |
| DB connector         | ConnectorX or ADBC | Polars-native Postgres ingestion                                                     |
| Rule/contract format | YAML               | Human-readable, diffable, versionable                                                |
| Scoping interface    | CLI + Streamlit    | Air-gapped/no-npm: in-process driver compiles the graph with its own checkpointer (agent-chat-ui dropped on `feat/bedrock-proxy-adapter`) |
| Testing              | pytest             | Standard                                                                             |

---

## Open Questions

- Which Postgres connector integrates most cleanly with Polars in practice? (ConnectorX vs ADBC)
- ~~Should the CLI be the only interface for v1, or is a minimal web UI worth scoping early?~~
  Answered: minimal web UI, localhost demo first (see Design Decisions)
- ~~Multi-tenancy: single user for now; if internal deployment, does the registry need
  access control per team or dataset?~~ Answered: deferred until the demo proves the
  workflow; decide with IT (see Design Decisions)
- How large are the real Postgres tables at work? Determines whether profiler sampling
  hooks are sufficient or SQL pushdown becomes necessary
