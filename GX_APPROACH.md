# Our approach to Great Expectations

## Version

Great Expectations **0.18.2**, the version data-flow uses. Every contract pins it (`gx_version: "0.18.2"`) and is rejected if it says anything else. We don't upgrade it without agreeing with data-flow first.

## What a check is

One **contract** per table: a YAML file holding a plain GX expectation suite, wrapped with the target table and an approval.

```yaml
contract_version: 1
gx_version: "0.18.2"
dataset: {schema: sales, table: orders}
approval: {approved_by: jane, approved_at: 2026-10-08T00:00:00Z}
suite:
  name: sales.orders
  expectations:
    - expectation_type: expect_column_values_to_not_be_null
      kwargs: {column: customer_id}
```

Expectation names and kwargs are GX's own. Nothing is translated or generated.

## Where checks are stored

- **Contracts:** `dags/<team>/contracts/<schema>.<table>.yaml` in the data-flow repo, committed to git. The YAML file is the only durable copy of the rules. We don't use GX's own suite or checkpoint stores.
- **Results:** in data-flow's GX validation store (`dataflow.ge_validation_store`), through `gx_get_context()`.

## How checks are written

1. The dq-agent scoping agent profiles the table and drafts a contract with the dataset owner.
2. The owner approves it, which adds the `approval` block.
3. The contract is copied into the team's `contracts/` folder in data-flow.

You can try a contract before it goes to data-flow with `notebooks/engine_gx_run.ipynb`, against Postgres or a local CSV/Parquet file.

## How checks run

- In **Airflow**, in data-flow. Nothing runs from dq-agent.
- A validation DAG (`dq_check_<table>_pipeline.py`) runs a `@task.virtualenv` task that calls `operators/dq_validation.py`.
- That module loads the contract, validates it, builds the suite and checkpoint in memory, and runs them against the Postgres table. GX turns the checks into SQL that runs inside the database.
- Each expectation has a severity in its `meta`. If an `error` expectation fails, the task raises and Airflow marks the run failed. If a `warning` expectation fails, it is logged and recorded in the validation result, and the run completes. An expectation without a severity is an `error`, and a check that could not run at all (a missing column, a regex Postgres rejects) always counts as an `error`.
- The DAG can run on a schedule, after an ingestion task, or when the table's Airflow Asset is updated.

## What a contract may contain

`dataflow_kit/operators/dq_validation.py` defines this for data-flow. dq-agent applies the same rules through its own copy, `src/dq_agent/gx_contract.py`, because neither side may import the other; `tests_kit/test_kit_parity.py` fails if the two disagree on any contract `tests/test_gx_contract.py` tries.

- **Allowed expectations:** a short allowlist (`ALLOWED_EXPECTATIONS`, 16 expectations that run in Postgres). Any other expectation is rejected.
- **Allowed kwargs:** what GX accepts for that expectation, minus `FORBIDDEN_KWARGS` (e.g. `row_condition`, `result_format`), which would change what data is checked or how results are reported.
- **Values:** checked by GX's own `validate_configuration`, plus a few rules of ours (no empty value sets, no `_between` check without bounds, at least two columns for multi-column checks, a valid `ignore_row_if`, only a relative-date `$PARAMETER`).
- **Meta:** only `severity` (`error` or `warning`) and the dashboard `notes`.
- **Approval:** required. A contract without it never runs.
- **Connection strings:** never in the contract. They come from the Airflow environment.

New expectation types are proposed through dq-agent's creative mode (`proposals/`) and only added to the allowlist after developer review.

## GX vs our code

| GX does | We add |
| --- | --- |
| The expectations and running them (SQL in Postgres) | The contract format and the approval gate |
| Each expectation's accepted kwargs and value checks | The allowlist, forbidden kwargs and policy rules |
| Suites, checkpoints, results, the validation store | Building the suite in memory from the YAML on each run |
| `$PARAMETER` evaluation | Stamping each result with who approved the rule |
| | Severity, a short pass/fail summary with no row values, and failing the Airflow task on an `error` |
