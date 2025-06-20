from airflow import models

from airflow.providers.google.cloud.operators.bigquery_dts import (
    BigQueryCreateDataTransferOperator,
    BigQueryDeleteDataTransferConfigOperator,
    BigQueryDataTransferServiceStartTransferRunsOperator,
    # BigQueryDataTransferServiceTransferRunSensor
)

from airflow.utils.dates import days_ago
from datetime import timedelta
import json


PROJECT_ID = "cloud-ml-auto-solutions"
LOCATION = "US" # Data Transfer Service location (often matches dataset location)
DESTINATION_DATASET_ID = "xlml_bite_testresults"
DISPLAY_NAME = "Axlearn unit test result GCS to BigQuery Transfer"
GCS_BUCKET_SOURCE = "gs://ml-auto-solutions/output/sparsity_diffusion_devx/axlearn-unit-tests/test-results/*.csv" # Source path in GCS
DESTINATION_TABLE_NAME = "axlearn_unit_test_results"

PYTHON_TABLE_SCHEMA = [
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

# Convert the Python list of dictionaries to a JSON string
JSON_TABLE_SCHEMA_STRING = json.dumps(PYTHON_TABLE_SCHEMA)

with models.DAG(
    dag_id="bigquery_data_transfer_dag",
    start_date=days_ago(1),
    schedule_interval=timedelta(days=1), # Schedule the DAG to run daily
    catchup=False,
    tags=["bigquery", "dts", "data_transfer"],
) as dag:

    create_transfer_config = BigQueryCreateDataTransferOperator( ##
        task_id="create_gcs_to_bq_transfer_config",
        project_id=PROJECT_ID,
        location=LOCATION,
        transfer_config={
            "destination_dataset_id": DESTINATION_DATASET_ID,
            "display_name": DISPLAY_NAME,
            "data_source_id": "google_cloud_storage", # Source type
            "schedule_options": {"disable_auto_scheduling": False},
            "params": {
                "data_path_template": f"{GCS_BUCKET_SOURCE}", # Use wildcard for files
                "destination_table_name_template": DESTINATION_TABLE_NAME,
                "file_format": "CSV",
                "skip_leading_rows": "1", # If CSV has headers
                "field_delimiter": ",",
            },
            "schedule": "every 3 hours",
            "disabled": False,
        },
        gcp_conn_id="google_cloud_default",
    )

    delete_transfer_config = BigQueryDeleteDataTransferConfigOperator(
        task_id='delete_dts_config',
        transfer_config_id="6884eb0d-0000-2021-80e7-582429a9f21c", ##"6884eb0d-0000-2021-80e7-582429a9f21c", ##"687f306c-0000-207b-ab7b-582429a8d768",###'687ceda3-0000-2656-befb-582429a80b94', ##687f306c-0000-207b-ab7b-582429a8d768 # Replace with the actual ID
        project_id=PROJECT_ID,  # Replace with your GCP project ID
        # Optional: Specify region if your transfer config is regionalized
        # location_id='your_location',
    )

    ##Run to manually trigger the data transfer
    start_transfer_run = BigQueryDataTransferServiceStartTransferRunsOperator(
        task_id="start_manual_transfer_run",
        project_id=PROJECT_ID,
        location=LOCATION,
        # transfer_config_id="68780005-0000-2d2e-812f-582429ad9dec",
        transfer_config_id="{{ task_instance.xcom_pull('create_gcs_to_bq_transfer_config', key='return_value')['name'].split('/')[-1] }}", # Get config ID from previous task
        # Define the time range for the transfer run (e.g., for data loaded since last run)
        requested_time_range={
        # Start time: A fixed point in the past (e.g., a known epoch or a very early date)
        # Be careful with very long ranges, as it may process too much data.
            "start_time": "1970-01-01T00:00:00.000000Z", # Example: Unix epoch
            # End time: The actual timestamp when the task instance starts executing
            "end_time": "{{ ts }}", # {{ ts }} provides YYYY-MM-DDTHH:MM:SS.ffffffZ
        },
        gcp_conn_id="google_cloud_default",
    )

    delete_transfer_config >> create_transfer_config >> start_transfer_run

