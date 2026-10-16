# dq-agent

[![tests](https://github.com/JBarone90/dq-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/JBarone90/dq-agent/actions/workflows/tests.yml)


**dq-agent helps a dataset owner write data-quality checks without writing code.** You point it at your data and describe its business context; it profiles the data and helps you assemble a **contract** — a [Great Expectations](https://greatexpectations.io/) 0.18.2 suite for one Postgres table. You approve the contract once, and from then on an Airflow task runs it **deterministically** on every pipeline run.

A large language model helps you _author_ the contract through a conversation, but it **never touches your data at execution time**. Approved checks run as Great Expectations expectations, not prompts. The LLM is a reasoning layer for scoping, not an execution engine.

The repo has two parts:

- **dq-agent** (`src/dq_agent/`): the scoping side — profiler, expectation catalogue, conversational agent and approval gate — driven from a terminal CLI (`scripts/scoping_cli.py`) or an optional Streamlit chat panel.
- **the data-flow kit** (`dataflow_kit/`): the run side, staged here and copied into the `data-flow` Airflow repo — one utility module, one example validation DAG, and the contracts folder.

> **Why Great Expectations?** At work, pipelines run only in Airflow and the department already uses Great Expectations (0.18.2) there, with results persisted to a validation store and shown on the monitoring dashboard. dq-agent therefore does not run checks itself: its product is a GX-native contract that the shared data-flow task runs. The differentiator is the step _before_ execution — turning a dataset owner's knowledge into a reviewed suite through conversation.

## Status

| Capability                                      | State                          |
| ----------------------------------------------- | ------------------------------ |
| Deterministic profiler (CSV + Postgres)         | **Done** — tested, usable      |
| Expectation catalogue (16 GX expectations)      | **Done** — tested              |
| Scoping agent + human approval gate             | **Working dev demo** (Phase 3) |
| data-flow kit (utility module + example DAG)    | **Pilot** — to be tried in data-flow |
| Creative mode — proposals for unlisted GX expectations | **Working dev demo** (Phase 4) |

The Polars execution engine the project started with is retired; it is preserved on the `feat/bedrock-proxy-adapter` branch. See [ACTION_PLAN.md](ACTION_PLAN.md) for the full roadmap.

## Contents

### Guide

- [How it works](#how-it-works)
- [Quickstart](#quickstart)
- [The scoping agent](#the-scoping-agent)
- [Project structure](#project-structure)

### Concepts

- [The run side](#the-run-side)
- [The agentic layer](#the-agentic-layer) (incl. [creative mode](#creative-mode))

### For developers

- [Development & testing](#development--testing)
- [Adding an expectation](#adding-an-expectation)
- [Design principles](#design-principles)

## How it works

The tool has two phases that never overlap in time. The only thing they share is the **contract YAML**: one phase produces it, the other consumes it.

```text
SCOPING TIME — once per table, human in the loop, LLM allowed (dq-agent)
  load data  →  profile (+ redact)  →  converse & propose expectations  →  human approval  →  <schema>.<table>.yaml

RUN TIME — every pipeline run, deterministic, no LLM anywhere (data-flow / Airflow)
  contract YAML  →  validate against allowlist  →  in-memory GX suite + checkpoint  →  pass / fail (fail raises)
```

```mermaid
flowchart LR
    subgraph S["Scoping time · dq-agent · human + LLM"]
        direction LR
        A[load data] --> B[profile + redact] --> C[converse & propose] --> D[human approval]
    end
    D --> E[(schema.table.yaml)]
    subgraph R["Run time · data-flow Airflow task · deterministic"]
        direction LR
        F[load + validate contract] --> G[GX checkpoint on the table] --> H[pass / fail]
    end
    E --> F
```

The profiler informs which expectations to _propose_; Great Expectations _executes_ what was approved. The LLM only ever sees a **redacted** profiler report (aggregates and hints, no raw cell values).

## Quickstart

**Requirements:** Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/). `pyproject.toml` and `uv.lock` target the work environment (its package mirrors and the internal `dwutils`); elsewhere, sync with `uv sync --all-extras --no-install-package dwutils --index-strategy unsafe-best-match` and run with `uv run --no-sync`.

```bash
uv sync --extra agents
uv run python scripts/scoping_cli.py
```

The repo ships a dirty-by-design dataset (`data/synthetic/orders.csv` — 20 order rows with known issues: null customer ids, a duplicate order id, a negative amount, a malformed email, an unexpected status, a stale date) and the contract that covers it (`contracts/examples/synthetic.orders.yaml`; the kit keeps the same one for data-flow). Loading and validating a contract is what the data-flow task does first, and dq-agent applies the same checks:

```python
from dq_agent.gx_contract import load_contract

contract = load_contract("contracts/examples/synthetic.orders.yaml")  # raises ContractError if invalid
```

Running it needs Great Expectations 0.18.2 and a Postgres table, so it happens in data-flow — see [The run side](#the-run-side).


## The scoping agent

Phase 3 wraps the scoping workflow in a single LangGraph agent (`src/dq_agent/agents/scoping.py`). It converses with the dataset owner, profiles the dataset (`profile_dataset` — redacted report only, no raw cell values ever reach the LLM), browses the expectation catalogue (`list_rules`), and proposes a draft contract (`propose_contract`, validated against the same allowlist data-flow uses). Approval is a LangGraph `interrupt()`: the graph pauses and presents a plain-English summary of the proposed checks (`report.describe_contract`) alongside the full YAML; a human approves / edits / rejects, and only an approved contract is stamped (`approval.approved_at`, `approval.approved_by`) and persisted to `contracts/<schema>.<table>.yaml`, ready to commit to the team folder's `contracts/` in data-flow.

data-flow enforces the gate at run time: its loader rejects a contract without an `approval` block.

### Running the chat interface

> **Branch note (`feat/bedrock-proxy-adapter`).** This branch targets an air-gapped work
> environment with no npm mirror, so it does **not** use agent-chat-ui or a `langgraph dev`
> server. The interface is a local, in-process driver instead: a terminal CLI today, and an
> optional Streamlit chat panel for non-technical owners. The default model is the internal
> Bedrock proxy (`DeptBedrockChat`), not Gemini.

**1. Install the agent extra.**

```bash
uv sync --extra agents
```

The default model is `DeptBedrockChat` (`src/dq_agent/agents/bedrock_chat.py`), which reaches Anthropic-on-Bedrock through the internal `dwutils.bedrock` proxy — no API key, no internet. `dwutils` is an internal package assumed importable in the work environment; set `DQ_AGENT_MODEL` to pick a Bedrock model id (defaults to `eu.anthropic.claude-sonnet-5`). The `model` argument to `build_graph` stays injectable, so a different LangChain model can be substituted in tests or off-network experiments without code changes.

**2. Run the CLI.** From the repo root:

```bash
uv run python scripts/scoping_cli.py                       # in-memory session
uv run python scripts/scoping_cli.py --db scoping.sqlite   # durable, resumable thread
```

Then chat: point the agent at `data/synthetic/orders.csv`, describe the business context, and iterate on its proposal. When the agent proposes a contract it pauses at the approval gate and prints the plain-English review; type `approve` to accept (which writes the contract to `contracts/<schema>.<table>.yaml`) or type feedback to have it revise. `--db` keeps the thread on disk so you can stop and resume the same `--thread` later.

**Why the CLI compiles the graph with a checkpointer.** The approval gate is a LangGraph `interrupt()`: it pauses the run by persisting it to a checkpointer (keyed by `thread_id`) and resumes by looking it back up. Without a checkpointer the gate can never complete and no contract is produced — so the driver passes one in (`MemorySaver` by default, `SqliteSaver` with `--db`). On `main`, the `langgraph dev` server supplied this implicitly; here the driver owns it.

The approval interrupt uses the project's own HITL contract (`action_requests` + `review_configs`); `agents/gate.py` holds that protocol for both drivers; its `unwrap_decision()` normalizes the resume payload, including a free-text reply, so a plain terminal can drive the gate without rendered buttons.

**3. Or run the visual chat panel (Streamlit), for non-technical owners.** Same in-process graph and interrupt/resume contract as the CLI, rendered as a chat UI — npm-free (Streamlit's frontend ships inside its Python wheel):

```bash
uv run --extra ui streamlit run app/scoping_app.py
```

It shows the conversation with a toggle to reveal the agent's tool calls, streams replies as they are produced, and renders the approval gate as approve / edit / reject controls (edit lets the owner amend the contract YAML, severities included, before approving). The sidebar shows the session's token usage; the adapter does not report cost. Files can be scoped only from under `data/`.

## Project structure

```text
dq-agent/
├── src/dq_agent/
│   ├── gx_contract.py  # the kit's rules, copied for the agent (allowlist, policy, severity); tested equal to the kit
│   ├── registry.py     # builds the agent's catalogue from GX metadata + our notes
│   ├── models.py       # Contract — the GX envelope, validated by gx_contract
│   ├── profiler.py     # dataset profiler + redact() — scoping-time facts for the LLM
│   ├── connectors.py   # load CSV / Parquet / Postgres into a Polars DataFrame for profiling
│   ├── report.py       # describe_contract() — contract -> plain-English approval card
│   ├── proposals.py    # creative mode: unlisted GX expectations, proposal files
│   └── agents/         # LangGraph scoping agent + human approval gate; gate.py is the driver protocol
├── app/scoping_app.py  # Streamlit chat driver
├── registry/expectations.yaml  # our notes per allowed expectation (tags, summary, caveats)
├── dataflow_kit/       # mirrors a data-flow team folder (dags/<team>/), copied over at work
│   ├── dq_check_example_pipeline.py  # example validation-only DAG
│   ├── operators/dq_validation.py    # allowlist, load/validate contract, run it as an in-memory GX checkpoint
│   └── contracts/        # one contract per table: <schema>.<table>.yaml
├── notebooks/engine_gx_run.ipynb  # run a contract with GX 0.18.2 before it goes to data-flow
├── contracts/          # approved contracts the scoping agent persists land here (git-ignored)
├── proposals/          # creative-mode proposals to allow more GX expectations — reviewed, never executed
├── data/synthetic/     # dirty-by-design test dataset (orders.csv)
└── tests/
```

**Polars is the profiling representation.** Connectors load into a Polars DataFrame before profiling; execution happens in Great Expectations, inside the database.


## Concepts

The toolkit is two layers. The **run side** executes checks with no LLM involved, inside Airflow. The **scoping layer** sits on top and is how an LLM helps a human author a contract for the run side.

### The run side

#### Expectation

**The smallest unit — one check.** A Great Expectations expectation, used by its GX name and kwargs, e.g. `expect_column_values_to_not_be_null` with `{column: customer_id}`. Sixteen are allowed today, covering completeness, uniqueness (single and compound), allowed and forbidden values, ranges, text format and length, column type, comparisons between columns, row count, the column set, freshness and stale rows.

#### Allowlist and registry

**Which expectations and kwargs a contract may use.** `dataflow_kit/operators/dq_validation.py` is the enforcement point in data-flow; the agent applies the same rules through its own copy, `src/dq_agent/gx_contract.py` (neither may import the other, and `tests_kit/test_kit_parity.py` fails if they disagree). Both lean on GX itself: `ALLOWED_EXPECTATIONS` lists the permitted expectation names; the kwargs each one may take are GX's own domain and success keys minus a few forbidden ones (`row_condition`, `result_format`, …); GX's `validate_configuration` checks the values; and a handful of policy rules cover what GX lets through (unknown kwargs, an empty `value_set`, a missing `regex`, a range with no bound, fewer than two columns for a multi-column check, an invalid `ignore_row_if`, any `$PARAMETER` other than the freshness form). A contract therefore cannot smuggle in arbitrary classes, actions or code.

The agent's catalogue is built from the same source at startup: for each allowed name, `registry.py` reads the expectation's docstring (description and per-kwarg documentation), its kwargs and defaults from GX. `registry/expectations.yaml` adds only what GX cannot know: tags, a plain-English `summary` for the approval gate, and usage notes such as "regex is evaluated by Postgres, not Python".

#### Contract

**A GX suite for one table, approved by a human.** It is the boundary object — the one artifact the scoping layer produces and data-flow consumes:

```yaml
contract_version: 1
gx_version: "0.18.2"
dataset: { schema: synthetic, table: orders }
approval: { approved_at: 2026-10-08T00:00:00Z, approved_by: jacopo } # data-flow refuses to run without it
suite:
  name: synthetic.orders
  expectations:
    - expectation_type: expect_column_values_to_not_be_null
      kwargs: { column: customer_id }
      meta:
        severity: error # a failure stops the pipeline run
        notes: { format: string, content: Every order must belong to a customer }
    - expectation_type: expect_column_max_to_be_between # freshness: the newest row is recent, evaluated at run time
      kwargs: { column: created_at, min_value: { $PARAMETER: now() - timedelta(days=365) } }
      meta: { severity: warning } # a failure is recorded; the run completes
```

Each expectation's `meta.severity` says what a failure does: `error` fails the data-flow task, `warning` is logged and recorded and the run completes. Without one it is an `error`, and a check that could not run at all (a missing column, a regex Postgres rejects) always counts as an `error`. The owner decides severity at the approval gate, which shows it next to every check.

Contracts never hold connection strings; the task reads the connection from the environment.

#### The data-flow task

`dataflow_kit/dq_check_example_pipeline.py` is a validation-only DAG. Its `@task.virtualenv` task calls `gx_get_context()` (data-flow's own `validation_utils`), `load_contract(contract_path(schema, table))` and `run_contract(...)`, which builds the suite and checkpoint in memory, runs them against the table and raises `DataQualityCheckFailed` when an `error` expectation fails — so Airflow marks the run failed. Failed `warning` expectations are logged and stay in the stored validation result. To validate after an ingestion pipeline instead, chain the same task after its sync (`sync_task >> check(...)`).

Before a contract goes to data-flow, `notebooks/engine_gx_run.ipynb` runs it with GX 0.18.2: against the real table through the same `run_contract` (the test that counts), or against a local CSV/Parquet file as a quick smoke test. It takes an approved contract file or the draft YAML the agent shows at the approval step, applies data-flow's checks first, and stores nothing.

### The agentic layer

These exist only at scoping time, to help a human and an LLM produce a good contract for the deterministic layer above. The LLM appears only here — and even here it never sees raw data.

#### Profiler

**The agent's window into the dataset.** The profiler (`profiler.py`) is the main tool the agent uses to reason about data it cannot see directly. `profile()` produces a structured report — per-column stats, table stats, and semantic hints — that tells the agent _which expectations are worth proposing_: a column hinted `email` suggests `expect_column_values_to_match_regex`; a column with no nulls that the owner says is mandatory suggests `expect_column_values_to_not_be_null`. Per-column stats include null rate, uniqueness, min/max, and distribution; semantic hints cover `email`, `phone`, `id`, and `date`. It is pure, deterministic code; the LLM consults its output, it does not run it.

#### Redaction

**The data-protection boundary between the profiler and the LLM.** A full profile can contain raw cell values (e.g. the most common values in a column), which must never be sent to a model. `redact()` returns a copy that is safe to share: raw value _examples_ (top values) are dropped, while bounded aggregates (null rate, uniqueness, numeric/temporal min/max) are kept — enough signal to propose rules, no raw data. The redacted `email` column, for instance:

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

Redaction is enforced in the agent's tooling: the `profile_dataset` tool always returns the redacted report, so a raw value cannot reach the model even by accident.

#### Creative mode

**Proposals to allow more GX expectations.** When an owner needs a check that no allowed expectation covers, the agent can look through the expectations Great Expectations 0.18.2 ships but the allowlist does not include (`proposals.py`): the diff between GX's registry (53 core expectations) and `ALLOWED_EXPECTATIONS`, each with its docstring summary and a Postgres-support hint read from GX's metric registry. It never invents a check of its own, and a proposal never enters a contract.

How it plays out in a conversation:

1. The owner asks for something nothing allowed covers, e.g. "the average order should stay between 10 and 200".
2. The agent says so, then calls `list_unlisted_expectations`. It finds `expect_column_mean_to_be_between` (Postgres: supported) and explains it to the owner.
3. If the owner wants it, the agent calls `propose_new_expectation`, which validates the example kwargs with the same GX + policy checks contracts get and writes `proposals/2026-10-08_expect_column_mean_to_be_between_<schema>.<table>.yaml`:

```yaml
status: proposed
proposed_at: '2026-10-08T14:02:11Z'
dataset: synthetic.orders
expectation_type: expect_column_mean_to_be_between
gx_description: Expect the column mean to be between a minimum value and a maximum value (inclusive).
sql_support: supported
rationale: the average order is between 10 and 200
example_kwargs: {column: amount, min_value: 10, max_value: 200}
example_pass: a day of typical orders   # invented examples — the agent never sees real values
example_fail: a day where a pricing bug zeroes most amounts
```

4. The agent tells the owner the check is not in the contract yet and carries on with the contract without it.
5. A developer reviews the file and promotes it: add the name to `ALLOWED_EXPECTATIONS` in the kit and in `gx_contract.py` (plus notes in `registry/expectations.yaml`), run it once against a real table, and set `status: promoted` — see [Adding an expectation](#adding-an-expectation).

The Postgres-support hint is only a hint: "supported" means every metric the expectation declares has a SQL implementation in GX; "unverified" can still work, because some expectations (e.g. `expect_column_values_to_be_in_type_list`) choose their metrics at run time. The real-table run at promotion decides.

## For developers

### Development & testing

Install the optional dependency groups you need:

| Command                    | Adds                                                    |
| -------------------------- | ------------------------------------------------------- |
| `uv sync`                  | core: Polars, Pydantic, PyYAML, `dwutils`, plus the `dev` group (pytest, pytest-cov, ruff, black, mypy, ipykernel) |
| `uv sync --extra postgres` | ConnectorX (Postgres connector), psycopg, SQLAlchemy 1.4 + psycopg2 (GX's Postgres driver) |
| `uv sync --extra agents`   | LangGraph + LangChain + SQLite checkpointer + Great Expectations 0.18.2 (the scoping agent, CLI driver and contract validation) |
| `uv sync --extra ui`       | Streamlit (the chat app) |

Run the suite:

```bash
uv run pytest                                  # everything the installed extras allow
uv run pytest tests/test_gx_contract.py        # one module
uv run pytest --cov                            # with coverage of src/dq_agent, app and scripts (.coveragerc)
```

Every test module skips cleanly when what it needs is missing, so the suite runs with any combination of extras: with core only, the profiler and connector tests run and the rest skip. With every extra installed, coverage is about 98%.

| Tests | Needs | What they cover |
| ----- | ----- | --------------- |
| `test_gx_contract.py` | `agents` | the allowlist and policy on a corpus of ~55 contracts (each with its expected verdict), loading files, severity on real GX runs (pandas, and SQL through SQLite), freshness vs stale rows |
| `test_models.py`, `test_registry.py`, `test_report.py`, `test_proposals.py` | `agents` | the contract model, the catalogue built from GX, the approval card, creative mode |
| `test_agents.py`, `test_gate.py` | `agents` | every tool, the approval gate (approve / edit / reject, every resume shape, reopening a saved thread), range-bound confirmation |
| `test_bedrock_chat.py` | `agents` | the Bedrock adapter against a stand-in `dwutils`: request shape, tool calls, streaming, caching, usage, error hints |
| `test_cli.py`, `test_app.py` | `agents` (+ `ui`) | both drivers end to end with a scripted model; the app runs headless through Streamlit's `AppTest` |
| `test_profiler.py`, `test_connectors.py` | core (+ `postgres`) | profiling, redaction, loaders and SQL building, DSN parsing |
| `test_postgres.py` | `agents`, `postgres`, `DQ_TEST_POSTGRES_URI` | a real Postgres: the known issues found, Postgres regex, both `$PARAMETER` date forms, the type names GX accepts, the newer expectations, profiling a table through the agent's tool |
| `test_live_bedrock.py` | `agents`, `dwutils`, `BEDROCK_PROXY_URL` / `BEDROCK_TOKEN` | the live proxy: a plain call, tool calling, a round trip, streaming, prompt caching, and one scoping session from profile to approved contract. Marked `integration` |

Nothing under `tests/` touches the network or the data-flow kit: the suite runs offline (Great Expectations' usage statistics are switched off in `tests/conftest.py`). Two markers separate what needs more than the repo:

```bash
uv run pytest -m "not postgres and not integration"   # the repo alone, offline
uv run pytest -m postgres                              # where DQ_TEST_POSTGRES_URI reaches a database
uv run pytest -m integration                           # where the Bedrock proxy is reachable
```

Without their environment variables the `postgres` and `integration` tests skip, so a plain `uv run pytest` is safe anywhere.

**At work.** The two marked modules are switched on by environment variables:

```bash
# a database where you may create a schema: the tests create dq_agent_test_<random> and drop it
DQ_TEST_POSTGRES_URI=postgresql://user:pass@host:5432/dbname uv run pytest tests/test_postgres.py -v
# the live proxy (costs a little budget): BEDROCK_PROXY_URL / BEDROCK_TOKEN are usually already set
uv run pytest -m integration -v
```

At home, a throwaway container stands in for the database:

```bash
docker run -d --rm --name dq-test-pg -e POSTGRES_PASSWORD=dq -p 5433:5432 postgres:16
DQ_TEST_POSTGRES_URI=postgresql://postgres:dq@localhost:5433/postgres uv run --no-sync pytest tests/test_postgres.py
docker stop dq-test-pg
```

**The data-flow kit** is checked separately, in this checkout only: `uv run pytest tests_kit` asserts that `dataflow_kit/operators/dq_validation.py` and `gx_contract.py` agree on the allowlist, the kwargs and every contract in `tests/test_gx_contract.py`. It lives outside `tests/` because the work repo has no kit.

**CI** runs `tests` and `tests_kit` on Python 3.13 with every extra and a Postgres 16 service, on push to `main` and on every pull request, deselecting the `integration` tests (`.github/workflows/tests.yml`).

### Adding an expectation

1. Add its name to `ALLOWED_EXPECTATIONS` in `dataflow_kit/operators/dq_validation.py` **and** in `src/dq_agent/gx_contract.py` (with any kwarg our policy requires beyond GX, e.g. `regex`), and any new policy rule to `_policy_errors` in both. Its kwargs, types and documentation come from GX automatically. Add a contract exercising any new rule to `CONTRACTS` in `tests/test_gx_contract.py`, with its expected verdict in `VERDICTS`; `tests_kit/test_kit_parity.py` then fails until the two files agree.
2. Add an entry to `registry/expectations.yaml` (a test requires one for every allowed expectation; without it the approval gate would show the raw name and kwargs):

```yaml
expect_column_values_to_not_be_null:
  tags: [completeness]
  summary: "`{column}` must never be empty"
  notes: Use for columns the owner says are mandatory.
```

3. Run it once against a real table, then copy the updated `dq_validation.py` into data-flow.

You rarely start from scratch: when an owner needs a check the allowlist lacks, the agent looks through the GX expectations that are not allowed yet (`list_unlisted_expectations`) and files a proposal in `proposals/` with the owner's rationale and validated example kwargs (`propose_new_expectation`). Reviewing that file is step 0; set its `status` to `promoted` once the steps above are done.

Authoring standards live in `.claude/roles/rule-author.md`.

### Design principles

- **Registry first.** The quality of the expectation catalogue determines the quality of every contract the agent produces. Invest there.
- **Deterministic by default.** The LLM is a reasoning layer, not an execution layer. Approved checks run as Great Expectations expectations, not prompts.
- **Framework agnostic at the model layer.** LLM provider is a config value. Swap between Anthropic, OpenAI, or a local model without code changes.
- **GX-native, allowlisted contracts.** Contracts use Great Expectations 0.18.2 names and kwargs directly, inside a small envelope; a deterministic allowlist decides what may run. No translation layer, no generated code.
