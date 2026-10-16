# DQ Agent — Action Plan

dq-agent helps a person who owns a dataset, but does not write code, end up with data quality checks that run automatically in the department's pipelines. A conversation helps them write the checks; from then on the checks run with no AI involved. The product is the **approved contract**: a Great Expectations 0.18.2 suite for one Postgres table.

The LLM earns its place in two places only: **scoping conversations** (interpreting business context that statistics alone can't capture) and **creative mode** (proposing that an unlisted GX expectation be allowed). Everything else is deterministic code.

---

## Architecture Overview

Two components, connected only by the contract file and the rules that judge it. This repo is primarily the first; it holds a copy of the second, which lives in the data-flow repo. A change to either is checked for its effect on the other.

```text
dq-agent (this repo) — scoping time, once per table, human + LLM
  profile (CSV/Parquet/Postgres) → redact → converse → propose (allowlisted GX expectations,
  each with a severity) → human approval gate → contracts/<schema>.<table>.yaml
        │
        │  the contract YAML, plus the same allowlist and policy on both sides
        ▼
data-flow kit (dataflow_kit/, copied into data-flow) — run time, every pipeline run, no LLM
  load + validate contract → in-memory GX suite + checkpoint on the table →
  error failure fails the Airflow task, warning failure is logged and recorded
```

- **Shared rules, two copies.** `dataflow_kit/operators/dq_validation.py` and `src/dq_agent/gx_contract.py` hold the same allowlist (16 expectations), forbidden kwargs, policy and severity rules. Neither imports the other; `tests_kit/test_kit_parity.py` fails if they disagree (run in this checkout; the work repo has no kit).
- **Single scoping agent.** One LangGraph agent with tools (`profile_dataset`, `profile_table`, `list_rules`, `propose_contract`, `request_approval`, plus the two creative-mode tools); the approval gate is an `interrupt()`. Sub-agents were planned and not needed.
- **Drivers.** A terminal CLI and a Streamlit app, both in-process with their own checkpointer, both through `agents/gate.py`. The model is `DeptBedrockChat` over the internal Bedrock proxy.

## Current state and next steps (2026-10-16)

Done: profiler and connectors, the GX-native contract and allowlist, severity, the scoping agent and gate, both drivers (with streaming and token usage), creative mode, and the kit with an example DAG. Tests run anywhere; the Postgres and live-proxy tests switch on at work (see README, Development & testing).

Next:
- [ ] Pilot in data-flow: copy the kit, run the example DAG on `synthetic.orders` and then on a real table (one passing run, one deliberately failing run), and confirm where results are stored (see "Open items for the data-flow pilot" below)
- [ ] Run the Postgres tests at work (`DQ_TEST_POSTGRES_URI`) against the department's Postgres version. They pass on Postgres 16 (2026-10-16): Postgres regex with POSIX classes, both `$PARAMETER` date bounds on a DATE column, and the `type_list` names (SQLAlchemy's, exact case: `INTEGER`, `DOUBLE_PRECISION`, `TEXT`, `DATE`)
- [ ] Decide whether to replace the removed evaluation harness: nothing now checks that a prompt or model change still catches the synthetic dataset's known issues, apart from the opt-in live session test
- [ ] Decide who approves: today the approver is the person at the keyboard (`DQ_AGENT_APPROVER` or the OS user); a sign-off by someone other than the owner is not covered
- [ ] Freshness (decided 2026-10-16): "the newest row is recent", `expect_column_max_to_be_between` with the `$PARAMETER` lower bound; `expect_column_min_to_be_between` with the same bound is the separate "no stale rows" check

## History

The phases below are the original build log, kept for the reasoning behind each decision. Items that mention the Polars execution engine, `RuleResult`, the run report, the `columns` snapshot or `registry/rules/` describe the design before `feat/gx-contracts` (2026-10-08), which replaced the engine with Great Expectations in data-flow; the engine survives on `feat/bedrock-proxy-adapter`.

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
- [x] Streamlit chat panel for non-technical owners (`app/scoping_app.py`, `ui` extra): same in-process graph + checkpointer pattern as the CLI, npm-free. Transcript with a tool-call toggle, profile table, approve / edit / reject gate (edit exposes the contract YAML, keyed to the draft), streamed replies, token usage, resumable threads. Auth, deployment and multi-user are deferred (decide with IT later)

**Exit criteria:** user points at a dataset, describes its business context, receives a proposed
rule contract, approves it, and a YAML contract file is produced that the Phase 1 engine can
execute directly.

---

### Phase 4 — Creative Mode

_Novel rule proposals for patterns outside the registry._

- [x] During scoping, agent identifies dataset characteristics with no matching registry rule
- [x] Agent drafts a rule specification: name, description, rationale, suggested parameters,
      example pass/fail cases — saved as a proposal file, not an executable rule
- [x] Proposal format is a lightweight config that a developer can review and promote to
      the registry with minimal effort
- [x] Track proposals separately from approved rules (different directory, clear status field)
- [x] Do not execute proposal rules — they are hypotheses until promoted
- [ ] Exit criterion not yet shown: needs a live-model scoping session where the owner asks for a check outside the allowlist

Implemented on `feat/gx-contracts` (2026-10-08) with GX as the source of candidates rather than free-form rule specs: the agent can only propose expectations GX 0.18.2 already ships. `list_unlisted_expectations` is the diff between GX's expectation registry (53 core expectations) and `ALLOWED_EXPECTATIONS`, each with its docstring summary and a Postgres-support hint read from GX's metric registry ("supported" when every declared metric has a SQLAlchemy provider, else "unverified" — a hint only, since some expectations pick metrics at run time). `propose_new_expectation` validates the example kwargs with the same GX + policy checks contracts get and writes `proposals/<date>_<expectation>_<schema>.<table>.yaml` (`status: proposed`, rationale, example kwargs, invented pass/fail examples). Promotion = add the name to the allowlist, run once on a real table, set `status: promoted`. Custom (non-GX) expectations remain out of scope.

**Exit criteria:** agent surfaces at least one novel rule proposal during a scoping session on
a dataset with an unusual pattern; proposal file is readable and actionable by a developer.

### Deferred — Registry-driven contract summaries

`report.describe_contract()` renders a contract as plain English for the approval gate, but it does so with a per-`rule_id` `if/elif` chain. That couples the renderer to the rule catalogue: adding a rule to `registry/rules/` requires editing `report.py` to phrase it well, which cuts against the registry invariant ("adding a rule is just YAML + a function, never a core-code change"). It degrades gracefully today (unknown rules fall back to the registry name), so this is a scalability cleanup, not a bug.

- [x] Move phrasing into the rule YAML as an optional `summary` template field (e.g. `` "`{column}` must never be empty" ``); `describe_contract` renders it generically via `str.format(**params)`, falling back to rule name + params when absent. New rules then ship their own description with no `report.py` edit.
- [x] Decide how to handle the few rules whose phrasing is conditional on a value (`null_check` at `max_null_rate: 0` vs. > 0; `range_check` one bound vs. two) — either accept blunter unconditional wording or keep a tiny override map only for those.
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
- ~~**Own the contract format.**~~ Superseded 2026-10-08: contracts are Great Expectations
  0.18.2-native (see the design decision below).

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


**Execution moves to Great Expectations in data-flow (2026-10-08, `feat/gx-contracts`).** At work, pipelines can only run as Airflow DAGs in the `data-flow` repo, and the department implicitly standardises on Great Expectations, pinned to 0.18.2, with validation results persisted through a GX validation store and shown on the pipeline monitoring dashboard. Running a parallel Polars engine would duplicate that, so the engine and rule functions are retired (kept on `feat/bedrock-proxy-adapter`) and the contract becomes a GX-native suite in a small envelope (`contract_version`, `gx_version`, `dataset.schema/table`, `approval`, `suite`). This reverses the "own the contract format" constraint. A data quality check in data-flow has three parts, two constant and one per table:
- **constant:** `dq_validation.py` — allowlist, contract loading/validation, in-memory suite + checkpoint, raise on failure;
- **constant:** an Airflow DAG — the pilot is a standalone validation-only DAG; integrating after a sync task (`sync >> check`) comes later;
- **per table:** `contracts/<schema>.<table>.yaml`, produced by the scoping agent and approved by the owner.

Both constant parts are staged in `dataflow_kit/` and copied into data-flow by hand; the footprint there is kept deliberately minimal (no scripts, data-flow-side tests postponed). The allowlist in `dq_validation.py` is the single definition of validity: dq-agent's `Contract.problems()` calls it. A first version also kept a hand-written per-rule catalogue in `registry/rules/` under a parity test; that duplication was removed the same day — permitted kwargs, value checks and the agent-facing descriptions are now read from GX 0.18.2 itself (`domain_keys`/`success_keys`, `validate_configuration`, docstrings), leaving only the list of allowed names, a few forbidden kwargs and policy rules, and `registry/expectations.yaml` (tags, plain-English summaries, caveats). GX 0.18.2 is therefore a dependency of the `agents` extra; it installs alongside LangChain/LangGraph and imports on Python 3.11–3.13. The approval is stamped into each expectation's `meta` at run time, because suite-level meta never reaches the stored validation result. All 8 previous rules have GX equivalents: null → `expect_column_values_to_not_be_null` (`mostly` = 1 − max null rate), unique, allowed values → `_be_in_set`, range → `_be_between`, regex → `_match_regex` (evaluated by Postgres, so Postgres regex syntax), row count → `expect_table_row_count_to_be_between`, type → `expect_column_values_to_be_in_type_list` (declared column type; text-encoded types go through regex), freshness → `expect_column_min_to_be_between` with `{"$PARAMETER": "now() - timedelta(days=N)"}` (verified to evaluate in 0.18.2; since 2026-10-16 freshness is `expect_column_max_to_be_between` with the same bound, and the min form is the stale-rows check). Dropped with the engine: per-rule severity (any failure fails the task), the `columns` schema-drift snapshot, and the run report (results go to the GX store and dashboard). The harness now scores proposals structurally by `(expectation_type, column)`.

Open items for the data-flow pilot:
- [ ] Confirm whether a manual-trigger DAG is acceptable, or whether the check should be scheduled on the table's Asset (needs the ingesting pipeline to declare it in `outlets`)
- [x] From the CompaniesHouseAccountsPipeline example (2026-10-09): Airflow 3 (`from airflow.sdk import dag, task, Asset`); team operators import as `dags.<team>.operators.<module>`; the check task pins `psycopg2-binary==2.9.10`; `max_active_runs=1`; ingestion tasks declare `outlets=[Asset("postgresql://datasets//<schema>/<table>")]`. Example DAG updated to match
- [ ] Run the example DAG on a local data-flow (`docker compose up`, then `airflow dags test`) against `synthetic.orders` loaded into the local datasets DB, before the real table: it verifies Postgres regex, `type_list` names and the write to `dataflow.ge_validation_store` with no real data
- [x] Layout decided (2026-10-09): everything lives in the existing team folder — `dags/<team>/dq_check_<table>_pipeline.py`, `operators/dq_validation.py`, `contracts/<schema>.<table>.yaml`; `dataflow_kit/` mirrors it
- [x] From the data-flow README (2026-10-09): DAG files must end in `_pipeline.py` to be picked up (example renamed); check tasks use `retries=5`; local dev runs Python 3.12; task code is pulled from GitHub at run time, so no deployment is needed for a new DAG or module
- [ ] Copy the kit, scope a real, well-known table with `profile_table`, run the example DAG: one passing run, one deliberately failing run
- [ ] Verify where the validation result is persisted: the data-flow README says `dataflow.ge_validation_store`, surfaced in the 'Data Validation' sheet of the pipeline monitoring dashboard, but the table was not visible from an interactive SQL session
- [ ] Run `notebooks/engine_gx_run.ipynb` against a real Postgres table: its Postgres path, Postgres regex and `type_list` names are untested so far (the local-file path is verified on `orders.csv`)
- [x] Tests for `dq_validation.py` (2026-10-16): `tests/test_gx_contract.py` pins the verdict on every contract in its corpus and runs contracts with GX on pandas and SQLite; `tests_kit/test_kit_parity.py` replays it against the kit; `tests/test_postgres.py` runs contracts on a real Postgres
- [x] Schema drift (2026-10-15): `expect_table_columns_to_match_set` is in the allowlist, replacing the dropped `columns` snapshot

**Severity and the extended allowlist (2026-10-15/16).** Per-rule severity is back, in GX form: each expectation's `meta.severity` is `error` (fails the task) or `warning` (logged and recorded, the run completes); missing means `error`, and a check that could not run is always an `error`. The owner chooses it at the approval gate. The allowlist grew to 16 Postgres-safe expectations (max-between, not-in-set, not-match-regex, value lengths, column pair A > B, compound unique, unique within record, columns match set), with policy rules for their multi-column kwargs. dq-agent stopped importing the kit: `src/dq_agent/gx_contract.py` carries the same rules for the agent, and `tests/test_gx_contract.py` holds the two to the same allowlist and the same verdict on every contract it tries, so both change together.

**Removed at work (2026-10-16).** The evaluation harness (`harness.py` and its live-LLM integration tests) and `scripts/explore_bedrock.py` were deleted; `BEDROCK_PROXY.md` keeps what the probe established. The notebook is now `engine_gx_run.ipynb` and runs on `gx_contract`. The project moved to Python 3.13 with the work package mirrors, and `dwutils` became a core dependency.
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
| Rule/contract format | YAML (GX 0.18.2-native) | Human-readable, diffable, versionable; runs unchanged in data-flow              |
| Execution            | Great Expectations 0.18.2 in Airflow (`data-flow`) | The department standard; replaced the Polars engine on `feat/gx-contracts` |
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
