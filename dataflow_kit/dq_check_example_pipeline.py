"""Example validation-only DAG: runs the approved data quality contract for one table.

It runs no ingestion. It loads `contracts/<schema>.<table>.yaml` and validates the table with Great Expectations through `dq_validation.run_contract`. A failed expectation raises, so Airflow marks the run failed.

Before using it at work:
- copy it into the team folder as `dags/<team>/dq_check_<table>_pipeline.py`, next to `operators/` and `contracts/`;
- set SCHEMA / TABLE to the pilot table (its contract must exist in contracts/);
- the file name must end in `_pipeline.py` for Airflow to notice it;
- replace `my_team` in the `dq_validation` import with the team folder name (data-flow DAGs import their operators as `dags.<team>.operators.<module>`);
- `schedule=None` means manual trigger only.

To validate after an existing pipeline instead, chain the same task after its sync: `sync_task >> check(...)`. Or keep this DAG separate and schedule it on the table's Airflow Asset, `schedule=[Asset("postgresql://datasets//<schema>/<table>")]`, so it runs whenever the ingesting pipeline updates the table — this works only if that pipeline declares the Asset in its `outlets`.
"""

import datetime

from airflow.sdk import dag, task

SCHEMA = "pilot_schema"  # TODO: the real pilot table
TABLE = "pilot_table"
DAG_ID = f"dq_check__{SCHEMA}__{TABLE}"


@task.virtualenv(
    requirements=[
        "psycopg2-binary==2.9.10",
        "sqlalchemy==1.4.40",
        "great-expectations==0.18.2",
    ],
    retries=5,
    system_site_packages=True,
)
def check(schema, table, dag_id, run_id):
    # Runs in a fresh virtualenv: every import must live inside the function.
    import os

    from dags.my_team.operators.dq_validation import load_table_contract, run_contract  # TODO: team folder name
    from dataflow.validation_utils import gx_get_context

    contract = load_table_contract(schema, table)
    run_contract(
        gx_get_context(),
        contract,
        connection_string=os.environ["AIRFLOW_CONN_DATASETS_DB"],
        checkpoint_name=dag_id,
        run_name=run_id,
    )


@dag(
    dag_id=DAG_ID,
    schedule=None,
    start_date=datetime.datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["data-quality"],
)
def dq_check_example():
    check(schema=SCHEMA, table=TABLE, dag_id=DAG_ID, run_id="{{ run_id }}")


dq_check_example()
