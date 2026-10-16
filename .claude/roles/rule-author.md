---
role: rule-author
applies_when: Editing registry/expectations.yaml, or ALLOWED_EXPECTATIONS / FORBIDDEN_KWARGS / the policy rules in dataflow_kit/operators/dq_validation.py or src/dq_agent/gx_contract.py
description: How to add or change an expectation the scoping agent may propose and data-flow may run
---

# Rule Author Role

## What a rule is now

A rule is a **Great Expectations 0.18.2 expectation** that contracts may use, under its GX name. dq-agent executes nothing: data-flow runs the contract with GX inside an Airflow task. What we write by hand is deliberately small, and everything else is read from GX:

| Defined by us | Read from GX 0.18.2 |
|---|---|
| `ALLOWED_EXPECTATIONS`: which names are permitted, plus kwargs our policy requires beyond GX | the kwargs each expectation takes (`domain_keys` + `success_keys`) |
| `FORBIDDEN_KWARGS`: GX kwargs a contract may never set | value checks (`validate_configuration`) |
| `_policy_errors`: rules for what GX lets through | the agent-facing description and per-kwarg docs (docstring) and defaults |
| `registry/expectations.yaml`: tags, plain-English `summary`, usage `notes` | |

Never restate kwargs, types or descriptions anywhere else — that is the duplication this design removed.

The rules live in two files that must stay identical in effect: `dataflow_kit/operators/dq_validation.py` (copied into data-flow) and `src/dq_agent/gx_contract.py` (what the agent validates with). Neither may import the other. Every change to the allowlist, `FORBIDDEN_KWARGS` or `_policy_errors` goes into both, in the same commit.

## Adding an expectation

Usually this starts from a file in `proposals/`, written by the agent's creative mode: the owner's rationale, the GX expectation, example kwargs already validated against GX, and a Postgres-support hint. Treat "unverified" support as unknown, not as unsupported.

1. Add the name to `ALLOWED_EXPECTATIONS` in both files. If GX silently defaults a kwarg to something vacuous (as `regex` defaults to match-everything), list it as required there.
2. Check what GX would let through for it (`ExpectationConfiguration` + `validate_configuration` in a REPL) and extend `_policy_errors` generically if needed, in both files — prefer a rule that applies to every expectation over a per-name branch.
3. Add `registry/expectations.yaml` tags, summary and notes (a test requires all three for every allowed expectation):
   - `summary` is shown to a non-engineer at the approval gate. Placeholders are kwarg names; a missing kwarg renders as "no limit", a list as comma-separated values, and `mostly < 1` adds "(for at least N% of rows)". Do not expose raw regexes or code.
   - `notes` is read by the LLM: when to use it, and backend caveats GX's docstring does not mention (e.g. regex is evaluated by Postgres, not Python).
4. Run it once against a real Postgres table with GX 0.18.2 before relying on it, then set the proposal's `status: promoted`.

## Constraints

- `dq_validation.py` is copied into data-flow: stdlib + PyYAML + GX only, never import `dq_agent`, and import GX inside functions so DAG parsing stays light. `gx_contract.py` has the same constraints and also must not import the rest of `dq_agent` (the registry and models import it).
- Expectation `meta` may hold only `severity` (`error` | `warning`, default `error`) and `notes`. A check that could not run always counts as an `error`.
- `FORBIDDEN_KWARGS` covers anything that changes which rows are checked (`row_condition`), how results are reported (`result_format`, `catch_exceptions`) or lets GX choose parameters (`auto`, `profiler_config`). Keep it that way.
- Evaluation parameters (`{"$PARAMETER": ...}`) are allowed only in the exact freshness form.

## Testing

- `tests/test_gx_contract.py`: add a contract for each new policy rule to `CONTRACTS`, with its expected error fragment in `VERDICTS`.
- `tests_kit/test_kit_parity.py` (`uv run pytest tests_kit`): the two rule files agree on that corpus; a mismatch means one file was not updated. The work repo has no kit, so it runs only in this checkout: run it before copying either file over. Never reference the kit from `tests/`.
- `tests/test_postgres.py`: anything Postgres decides (regex, type names, date bounds) gets a case here, run with `DQ_TEST_POSTGRES_URI`.
- `tests/test_registry.py`: the catalogue is built from GX, never offers a forbidden kwarg, and every allowed expectation has a summary.
- `tests/test_models.py`: add a `test_problems_reject` case for each new policy rule.
- `tests/test_report.py`: add a `describe_contract` assertion for a new summary.
