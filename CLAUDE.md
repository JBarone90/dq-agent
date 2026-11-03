# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                      # core deps + the dev group (pytest, ruff, mypy, ipykernel)
uv sync --extra postgres     # add ConnectorX, psycopg, SQLAlchemy 1.4 for the Postgres connector
uv sync --extra agents       # add LangGraph + LangChain (+ langchain-anthropic) + Great Expectations 0.18.2
uv sync --extra ui           # add Streamlit for app/scoping_app.py
uv run python -m ipykernel install --user --name dq-agent --display-name "Python (dq-agent)"   # once

uv run pytest                # run all tests (postgres / integration ones skip without their env)
uv run pytest -m "not postgres and not integration"   # the repo alone, offline
uv run pytest tests_kit      # the kit and gx_contract agree (this checkout only: no kit at work)
uv run pytest tests/path/to/test_file.py::test_name   # run a single test
uv run pytest --cov=src/dq_agent                      # run with coverage

uv run python scripts/scoping_cli.py                  # drive the scoping agent from the terminal
uv run python scripts/scoping_cli.py --db scoping.sqlite   # ...with a durable, resumable thread
# notebooks/engine_gx_run.ipynb (dq-agent kernel): run a contract with GX 0.18.2 (file or Postgres)
```

`pyproject.toml` and `uv.lock` are the work environment's: Python 3.13 and the internal package mirrors. Outside work the first mirror returns 403, so sync once with `uv sync --all-extras --index-strategy unsafe-best-match` and then run with `uv run --no-sync ...`. Do not edit the two files to make them install outside work, and lock at work: a lock made outside it with `unsafe-best-match` moves packages between mirrors.

## Architecture

The project separates the LLM reasoning layer from all data work. The LLM is intentionally limited to two roles: **scoping conversations** and **creative mode** (proposing novel rules). Everything else is deterministic code. Execution happens outside this tool: the department runs pipelines only in Airflow (`data-flow` repo) and validates with Great Expectations 0.18.2, so dq-agent's product is a GX-native contract that a shared data-flow task runs.

```text
registry/
  expectations.yaml  ← our notes per allowed expectation: tags, plain-English summary, usage caveats.
                       The rest of the agent's catalogue (descriptions, kwargs) is read from GX
dataflow_kit/        ← mirrors a data-flow team folder (dags/<team>/), copied over at work; must not import dq_agent
  dq_check_example_pipeline.py ← example validation-only Airflow DAG
  operators/dq_validation.py   ← ALLOWED_EXPECTATIONS (names) + policy, load/validate a contract, run it
                                 as an in-memory GX suite + checkpoint, raise on failure
  contracts/         ← one contract per table: <schema>.<table>.yaml
src/dq_agent/
  gx_contract.py     ← dq-agent's copy of the kit's rules (allowlist, policy, severity, run helpers);
                       must not import the kit or the rest of dq_agent; tests_kit/test_kit_parity.py holds it to the kit
  registry.py        ← builds the agent's catalogue from GX metadata + registry/expectations.yaml
  models.py          ← Contract: the GX envelope, with per-expectation severity; problems() delegates to gx_contract
  profiler.py        ← produces a structured JSON report (col stats, table stats, semantic hints);
                       redact() strips raw cell values before any report reaches an LLM
  connectors.py      ← adapters that load data into Polars DataFrames (CSV/Parquet, Postgres)
  agents/            ← LangGraph scoping agent; wraps profiler and registry as tools, human
                       approval gate as an interrupt(), persists approved contract YAML;
                       gate.py is the interrupt/resume protocol both drivers use
app/scoping_app.py   ← Streamlit driver (same graph + checkpointer pattern as the CLI)
notebooks/           ← engine_gx_run.ipynb: run a contract with GX before it goes to data-flow
contracts/           ← approved contracts the agent persists (<schema>.<table>.yaml, git-ignored)
proposals/           ← creative mode: proposals to allow an unlisted GX expectation (proposals.py),
                       awaiting developer review, never executed
data/synthetic/      ← dirty-by-design test dataset (committed); all other data excluded by .gitignore
```

### Two components, one repo

This repo is primarily **dq-agent** (the scoping side). `dataflow_kit/` is a copy of a **separate but connected component** that lives in the data-flow Airflow repo and is copied there by hand. The two meet only at the contract format and the rules that judge it. Whenever you change one, say what it means for the other: the contract shape, the allowlist and policy, severity semantics, the example contract, and which files have to be copied to data-flow. Kit-side changes are kept to what the agent side genuinely needs.

### Key design invariants

- **`tests/` runs at work, offline and without the kit.** The work repo has no `dataflow_kit/` and no internet. Nothing under `tests/` may import or mention the kit: tests target dq-agent's own code (`gx_contract`, `contracts/examples/`); the kit comparison lives in `tests_kit/`, run only in this checkout. Nothing may reach the network: `tests/conftest.py` switches off GX's usage statistics. Tests needing an optional extra skip without it; `postgres` tests (`DQ_TEST_POSTGRES_URI`) and `integration` tests (the Bedrock proxy) skip without their environment.

- **Polars is the profiling representation.** Connectors load into a Polars DataFrame before profiling. Execution is Great Expectations, inside the database, in data-flow.
- **Contracts are GX 0.18.2-native and allowlisted.** Expectation names and kwargs are used directly — no translation layer, no generated code, no `eval`. `dataflow_kit/operators/dq_validation.py` defines what data-flow accepts: `ALLOWED_EXPECTATIONS` lists the permitted names, the permitted kwargs are GX's own keys minus `FORBIDDEN_KWARGS`, values are checked by GX's `validate_configuration` plus a few policy rules, and `meta` holds only `severity` and `notes`. dq-agent does not import the kit: `src/dq_agent/gx_contract.py` carries the same rules, and `Contract.problems()` and the agent's catalogue (`registry.py`) derive from it, so never restate kwargs or types anywhere else. **Change the kit and `gx_contract.py` together** — `tests_kit/test_kit_parity.py` fails if they differ on the allowlist, the kwargs or any contract's verdict. Do not upgrade the GX version silently.
- **Severity decides what a failure does.** Each expectation's `meta.severity` is `error` (a failure fails the Airflow task) or `warning` (logged and recorded, the run completes). Missing means `error`, and a check that could not run is always an `error`. The owner chooses it at the approval gate; the agent only suggests.
- **Contracts gate execution.** No check runs without an approved contract: data-flow's loader rejects a contract without an `approval` block. Contracts never contain connection strings.
- **Keep the data-flow footprint minimal.** The kit is one utility module, one example DAG and the contracts folder, all inside the team folder. Do not add scripts or infrastructure there without asking.
- **Proposals are never executed.** `proposals/` contains hypotheses. Promote to the allowlist only after developer review.
- **LLM provider is a config value.** Agent code binds to LangChain's provider-agnostic interface. Swapping between Anthropic, OpenAI, or a local model requires no code changes.

## Roles

Role files in `.claude/roles/` contain task-specific coding criteria. Load the relevant file
before starting work in the corresponding area:

| When you are...                                      | Load this role file              |
| ---------------------------------------------------- | -------------------------------- |
| Editing `registry/expectations.yaml` or the allowlist/policy in `dataflow_kit/operators/dq_validation.py` or `src/dq_agent/gx_contract.py` | `.claude/roles/rule-author.md` |
| Editing the scoping agent's approval gate / interrupt payloads (`scoping.py`, `agents/gate.py`), or the CLI/Streamlit driver that consumes them | `.claude/roles/hitl-driver.md` |

---

### Build order

Phases are strictly sequential: Phase 1 (registry + engine) and Phase 2 (profiler + connectors) must be complete and tested before any LLM/agent work begins. The registry and engine are independently valuable — the agent is an interface layer on top.

---

## Branch: `feat/gx-contracts` (from `feat/bedrock-proxy-adapter`)

This branch replaces the Polars engine with GX-native contracts and the data-flow kit; the engine survives on `feat/bedrock-proxy-adapter`. Like its parent, it targets an **air-gapped work environment** and diverges from `main` on two axes — do not "fix" these back toward `main`:

- **Model.** The default chat model is LangChain's `ChatAnthropic`, built by `src/dq_agent/agents/llm.py` and pointed at the department's bedrock-proxy, which serves the Anthropic Messages API at `ANTHROPIC_BASE_URL` (token in `ANTHROPIC_AUTH_TOKEN`). The default is Sonnet 5.5 (`eu.anthropic.claude-sonnet-5-5`); `DQ_AGENT_MODEL` overrides it. There is no Gemini / `init_chat_model` path and no `GOOGLE_API_KEY`. `dwutils.bedrock` was removed on purpose (2026-11): do not reinstate it, pin an old `dwutils`, or rebuild a Converse adapter. `model` stays injectable in `build_graph` for tests.
- **Interface.** There is **no agent-chat-ui and no `langgraph dev` server** — the environment has no npm mirror. The interface is a **local, in-process driver** that compiles the graph itself and supplies its own checkpointer:
  - **CLI:** `scripts/scoping_cli.py` (the supported path today).
  - **Visual (for non-coders):** an optional Streamlit chat panel — same in-process graph + checkpointer pattern, npm-free.

**Proxy behaviour lives in `BEDROCK_PROXY.md`.** Credentials, model ids, what is verified to work through `ChatAnthropic`, prompt caching and the error shapes, checked against the live proxy. Read it before changing `llm.py`, and re-check it after any proxy upgrade (`uv run pytest -m integration -v` at work): the proxy is a pilot and has changed under us twice already.

**Checkpointer is mandatory for any driver.** The approval gate is a LangGraph `interrupt()`; without a checkpointer it cannot pause and resume, so the contract is never produced. On `main` the `langgraph dev` server supplied this implicitly — here each driver must pass one to `build_graph(checkpointer=...)` (`MemorySaver` for an ephemeral session, `SqliteSaver` for a resumable one). A driver loop must also distinguish a normal conversational turn (send a new message) from a pending interrupt (send `Command(resume=...)`) by checking `result.get("__interrupt__")`.
