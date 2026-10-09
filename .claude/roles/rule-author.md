---
role: rule-author
applies_when: Editing registry/expectations.yaml, or ALLOWED_EXPECTATIONS / FORBIDDEN_KWARGS / the policy rules in dataflow_kit/operators/dq_validation.py
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

## Adding an expectation

Usually this starts from a file in `proposals/`, written by the agent's creative mode: the owner's rationale, the GX expectation, example kwargs already validated against GX, and a Postgres-support hint. Treat "unverified" support as unknown, not as unsupported.

1. Add the name to `ALLOWED_EXPECTATIONS`. If GX silently defaults a kwarg to something vacuous (as `regex` defaults to match-everything), list it as required there.
2. Check what GX would let through for it (`ExpectationConfiguration` + `validate_configuration` in a REPL) and extend `_policy_errors` generically if needed — prefer a rule that applies to every expectation over a per-name branch.
3. Optionally add `registry/expectations.yaml` notes:
   - `summary` is shown to a non-engineer at the approval gate. Placeholders are kwarg names; a missing kwarg renders as "no limit", a list as comma-separated values, and `mostly < 1` adds "(for at least N% of rows)". Do not expose raw regexes or code.
   - `notes` is read by the LLM: when to use it, and backend caveats GX's docstring does not mention (e.g. regex is evaluated by Postgres, not Python).
4. Run it once against a real Postgres table with GX 0.18.2 before relying on it, then set the proposal's `status: promoted`.

## Constraints

- `dq_validation.py` is copied into data-flow: stdlib + PyYAML + GX only, never import `dq_agent`, and import GX inside functions so DAG parsing stays light.
- `FORBIDDEN_KWARGS` covers anything that changes which rows are checked (`row_condition`), how results are reported (`result_format`, `catch_exceptions`) or lets GX choose parameters (`auto`, `profiler_config`). Keep it that way.
- Evaluation parameters (`{"$PARAMETER": ...}`) are allowed only in the exact freshness form.

## Testing

- `tests/test_registry.py`: the catalogue is built from GX and never offers a forbidden kwarg.
- `tests/test_models.py`: add a `test_problems_reject` case for each new policy rule.
- `tests/test_report.py`: add a `describe_contract` assertion for a new summary.
