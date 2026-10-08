---
role: rule-author
applies_when: Writing or editing any file under registry/rules/, or the ALLOWED_EXPECTATIONS allowlist in dataflow_kit/dq_validation.py
description: How to add or change an expectation the scoping agent may propose and data-flow may run
---

# Rule Author Role

## What a rule is now

A rule is a **Great Expectations 0.18.2 expectation** that contracts may use, under its GX name (e.g. `expect_column_values_to_not_be_null`). dq-agent no longer executes anything: data-flow runs the contract with GX inside an Airflow task. A rule therefore lives in two places that must agree exactly:

1. **The allowlist** — `ALLOWED_EXPECTATIONS` in `dataflow_kit/dq_validation.py`. The enforcement point: data-flow rejects any expectation or kwarg not listed. Each kwarg maps to a small check function (`_string`, `_fraction`, `_non_empty_list`, ...).
2. **The catalogue** — `registry/rules/<expectation_type>.yaml`. What the agent reads to decide what to propose, and how the approval gate phrases it.

`test_registry_matches_the_data_flow_allowlist` fails until both list the same expectations, kwargs and required kwargs.

## Allowlist entry

```python
"expect_column_values_to_be_in_set": {
    "required": {"column": _string, "value_set": _non_empty_list},
    "optional": {"mostly": _fraction},
},
```

- Only kwargs GX 0.18.2 accepts for that expectation (check `get_expectation_impl(name).success_keys` and `domain_keys`). Never allow `row_condition`, `condition_parser`, `result_format`, `catch_exceptions`, `auto` or `profiler_config`.
- Allow `mostly` only where it means something (column map expectations); not on aggregate or schema checks.
- If an expectation needs at least one of several optional kwargs (a range), add it to `_ONE_OF`.
- Evaluation parameters (`{"$PARAMETER": ...}`) are allowed only through a dedicated check with an exact pattern, never as free expressions.
- The module must stay stdlib + PyYAML + GX and must never import `dq_agent`: it is copied into data-flow.

## Catalogue YAML

```yaml
id: expect_column_values_to_not_be_null
name: Not null
description: Fails if the column has null values. Set mostly below 1.0 to tolerate a share of nulls.
tags: [completeness]
kwargs:
  column: {type: str, required: true}
  mostly: {type: float, description: fraction of rows that must be non-null (default 1.0)}
summary: "`{column}` must never be empty"
```

- `description` is read by the LLM: say when to use the expectation and any backend caveat (e.g. regex is POSIX on Postgres).
- `summary` is shown to a non-engineer at the approval gate. Placeholders are kwarg names; a missing kwarg renders as "no limit", a list as comma-separated values, and `mostly < 1` adds "(for at least N% of rows)" automatically. Do not expose raw regexes or code in the summary.

## Testing

- The parity test covers the allowlist/catalogue match.
- Add a `describe_contract` assertion in `tests/test_report.py` for the new phrasing.
- Before promoting a new expectation, run it once against a real table with GX 0.18.2 (the allowlist cannot prove GX accepts a kwarg combination on Postgres).
