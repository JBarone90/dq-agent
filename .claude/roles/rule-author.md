---
role: rule-author
applies_when: Writing or editing any file under registry/rules/ or src/dq_agent/rules/
description: Coding criteria for authoring data quality rule functions in this project
---

# Rule Author Role

## What a rule is

A rule is a **pure function** that takes a Polars DataFrame and keyword parameters, checks one
data quality condition, and returns a `RuleResult`. Nothing else.

## Function signature

```python
def rule_name(df: pl.DataFrame, *, column: str, **kwargs) -> RuleResult:
```

- All parameters after `df` are keyword-only
- Parameter names must match the `parameters` keys in the rule's YAML definition exactly
- Return type is always `RuleResult` — never a plain dict or bool

## Invariants

- **No side effects.** Never write to disk, log, mutate the DataFrame, or call external services.
- **No LLM calls.** Rules are deterministic code only.
- **Polars expressions only.** Never iterate over rows. No pandas, no numpy loops.
- **Never raise.** Catch internal errors and surface them through `RuleResult.error` if needed.
- **Column existence is the caller's problem.** The engine validates column names before calling;
  rules can assume the column exists.

## RuleResult fields to populate

- `rule_id`: copy from the YAML `id`, hardcoded per function
- `passed`: bool — the single verdict
- `violation_rate`: fraction of rows that violated the rule (0.0–1.0) — always populate this
- `error`: populate only when the rule cannot run due to misconfiguration; set `passed=False`

## YAML definition (one file per rule)

```yaml
id: null_check
name: Null Check
description: Fails if null rate in a column exceeds the configured threshold.
tags: [completeness, any_type]
severity: error          # error | warning
parameters:
  column: {type: str, required: true}
  max_null_rate: {type: float, default: 0.0}
execution:
  module: dq_agent.rules.completeness
  function: null_check
```

## Testing

**Prefer conftest fixtures for standard cases.** The shared `clean_df` and `dirty_df` fixtures
in `conftest.py` represent the synthetic dataset and should be the default input for rule tests.
Use them for the happy path (passes on clean data, fails on dirty data).

**Use inline DataFrames only for rule-specific edge cases** — when the data is small (2–5 rows),
specific to one rule's boundary condition, and would be misleading as a shared fixture
(e.g. a DataFrame with all nulls used only to test `null_check` at 100% null rate).

Each rule must have tests that:
1. Assert `passed=True` on clean data (via fixture)
2. Assert `passed=False` on dirty data (via fixture or inline if fixture doesn't cover the case)
3. Check `metric` is the correct measured value
4. Cover at least one boundary/edge case inline

Tests live in `tests/rules/` mirroring the source structure.
