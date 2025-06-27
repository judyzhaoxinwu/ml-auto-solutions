from airflow import models
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCreateTableOperator,
    BigQueryCreateExternalTableOperator,
    BigQueryDeleteTableOperator
)
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
import json

# Define your project, dataset, and table details
PROJECT_ID = "cloud-ml-auto-solutions"
DATASET_ID = "xlml_bite_testresults"
TABLE_ID = "axlearn_unit_test_results"
TABLE_ID_TEST = "axlearn_unit_test_results_test"
TABLE_ID_TEMP = "axlearn_unit_test_results_temp"
TABLE_ID_EXTERNAL = "axlearn_unit_test_results_external"


GCS_BUCKET = "ml-auto-solutions"
GCS_FILE = "output/sparsity_diffusion_devx/axlearn-unit-tests/test-results/*.csv" # For external table or GCS to BQ load


# Define a schema for an empty table or external table
TABLE_SCHEMA = [
    {"name": "id", "type": "STRING", "mode": "NULLABLE"},
    {"name": "module", "type": "STRING", "mode": "NULLABLE"},
    {"name": "name", "type": "STRING", "mode": "NULLABLE"},
    {"name": "file", "type": "STRING", "mode": "NULLABLE"},
    {"name": "doc", "type": "STRING", "mode": "NULLABLE"},
    {"name": "markers", "type": "STRING", "mode": "NULLABLE"},
    {"name": "status", "type": "STRING", "mode": "NULLABLE"},
    {"name": "message", "type": "STRING", "mode": "NULLABLE"},
    {"name": "duration", "type": "STRING", "mode": "NULLABLE"},
    {"name": "platform", "type": "STRING", "mode": "NULLABLE"},
    {"name": "accelerator_type", "type": "STRING", "mode": "NULLABLE"},
    {"name": "datetime", "type": "TIMESTAMP", "mode": "NULLABLE"},
    {"name": "test_name", "type": "STRING", "mode": "NULLABLE"},
    {"name": "jax_version", "type": "STRING", "mode": "NULLABLE"},
]

JSON_TABLE_SCHEMA_STRING = json.dumps(TABLE_SCHEMA)


TABLE_RESOURCE = {
    "tableReference": {
        "projectId": PROJECT_ID,
        "datasetId": DATASET_ID,
        "tableId": TABLE_ID_TEST,
    },
    "schema": {
        "fields": TABLE_SCHEMA # Your schema goes here
    },
    "description": "Table for storing test results data.", # Optional description
}

with models.DAG(
    dag_id="bigquery_table_creation_dag",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=["bigquery", "table_creation"],
) as dag:
    with TaskGroup(
      group_id='bigquery_table', prefix_group_id=False
    ) as bigquery_test:

      create_table = BigQueryCreateTableOperator(
          task_id="create_bigquery_table",
          project_id=PROJECT_ID,
          dataset_id=DATASET_ID,
          table_id=TABLE_ID_TEST,
          # schema_fields=TABLE_SCHEMA,
          table_resource=TABLE_RESOURCE,
          gcp_conn_id="google_cloud_default",
          location="us-central1",
          if_exists="log",
      )

      delete_test_table = BigQueryDeleteTableOperator(
          task_id="delete_test_bigquery_table",
          deletion_dataset_table=f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID_TEST}",
          ignore_if_missing=True,
          location="US",
      )
      delete_external_table = BigQueryDeleteTableOperator(
          task_id="delete_external_bigquery_table",
          deletion_dataset_table=f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID_EXTERNAL}",
          ignore_if_missing=True,
          location="US",
      )
      delete_temp_table = BigQueryDeleteTableOperator(
          task_id="delete_temp_bigquery_table",
          deletion_dataset_table=f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID_TEMP}",
          ignore_if_missing=True,
          location="US",
      )

      # delete_external_table >>
      delete_external_table
      delete_temp_table
      delete_test_table
