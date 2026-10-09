"""Example validation-only DAG: runs the approved data quality contract for one table.

It runs no ingestion. It loads `dq_contracts/<schema>.<table>.yaml` and validates the table with Great Expectations through `dq_validation.run_contract`. A failed expectation raises, so Airflow marks the run failed.

Before using it at work:
- set SCHEMA / TABLE to the pilot table (its contract must exist in dq_contracts/);
- the file name must end in `_pipeline.py` for Airflow to notice it;
- adjust the `dataflow.dq_validation` import to wherever dq_validation.py is placed;
- `schedule=None` means manual trigger only; on Airflow < 2.4 the argument is `schedule_interval`.

To validate after an existing pipeline instead, chain the same task after its sync: `sync_task >> check(...)`.
"""

import datetime

from airflow.decorators import dag, task

SCHEMA = "pilot_schema"  # TODO: the real pilot table
TABLE = "pilot_table"
DAG_ID = f"dq_check__{SCHEMA}__{TABLE}"


@task.virtualenv(
    requirements=[
        "psycopg2-binary==2.8.6",
        "sqlalchemy==1.4.40",
        "great-expectations==0.18.2",
    ],
    retries=5,
    system_site_packages=True,
)
def check(schema, table, dag_id, run_id):
    # Runs in a fresh virtualenv: every import must live inside the function.
    import os

    from dataflow.dq_validation import load_table_contract, run_contract
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
    tags=["data-quality"],
)
def dq_check_example():
    check(schema=SCHEMA, table=TABLE, dag_id=DAG_ID, run_id="{{ run_id }}")


dq_check_example()
