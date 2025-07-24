"""
DAG to process unit test result CSV files from GCS, enrich the data,
and load it into a final BigQuery table.

This DAG is designed with a robust ELT pattern:
1.  Waits for a specific file to land in a GCS bucket in a different project.
2.  Ensures the BigQuery Dataset and final table exist.
3.  Creates a temporary external table pointing to the new file.
4.  Runs a transform-and-load query to enrich data from the filename and
    insert it into the final table.
5.  Cleans up by deleting the temporary table and archiving the source file.
"""
from __future__ import annotations

import datetime

from airflow.models.dag import DAG
from airflow.decorators import task

from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCreateEmptyDatasetOperator,
    BigQueryCreateEmptyTableOperator,
    BigQueryCreateExternalTableOperator,
    BigQueryDeleteTableOperator,
    BigQueryInsertJobOperator,
)
# from airflow.providers.google.cloud.operators.gcs_gcs import GCSMoveObjectOperator
from airflow.providers.google.cloud.transfers.gcs_to_gcs import GCSToGCSOperator

from airflow.providers.google.cloud.sensors.gcs import GCSObjectsWithPrefixExistenceSensor
from airflow.providers.google.cloud.operators.gcs import GCSDeleteObjectsOperator, GCSListObjectsOperator


# --- Configuration Variables ---
# Replace with your actual project, bucket, and dataset names
GCP_PROJECT_ID = "tpu-prod-env-one-vm" # The project where Composer and BigQuery live
GCS_BUCKET = "axlearn-arc-testing"
# The folder where new CSV files are dropped (no leading slash)
GCS_SOURCE_FOLDER = "testing/results/"
# The folder to move files to after processing
BIGQUERY_DATASET = "axlearn_arc_testing"
BIGQUERY_FINAL_TABLE = "unit_test_results"
# A unique name for the temporary table using the DAG's run_id
TEMP_EXTERNAL_TABLE = "temp_external_test_runs_{{ ts_nodash }}"
# TEMP_EXTERNAL_TABLE="temp_external_test_runs_20250722T162029"
# The filename pattern to look for.
# The connection ID for the external GCS project, configured in the Airflow UI
GCS_CONN_ID = "gcs_external_project_conn"
BIGQUERY_LOCATION = "US"  # The location for the BigQuery dataset


with DAG(
    dag_id="unit_test_results_to_bigquery",
    start_date=datetime.datetime(2025, 7, 21),
    schedule='0 */6 * * *',  # Set to None to trigger only when a file arrives
    catchup=False,
    tags=["bigquery", "gcs", "testing"],
    # This makes the TEMP_EXTERNAL_TABLE variable available in the SQL
    user_defined_macros={"TEMP_EXTERNAL_TABLE": TEMP_EXTERNAL_TABLE},
) as dag:
    # Task 1: Wait for a new CSV file to appear in the source folder.
    list_csv_files = GCSListObjectsOperator(
        task_id="list_csv_files",
        bucket=GCS_BUCKET,
        prefix=GCS_SOURCE_FOLDER,
        match_glob="**/unit-tests-*.csv",  # This pattern finds only CSV files
        gcp_conn_id=GCS_CONN_ID,
    )

    @task.short_circuit
    def check_if_files_exist(files_found: list) -> bool:
        """
        If files_found is not empty, returns True and allows downstream
        tasks to run. Otherwise, returns False and skips them.
        """
        print(f"Files found: {files_found}")
        return len(files_found) > 0

    check_task = check_if_files_exist(
        files_found=list_csv_files.output,
    )

    # # Task 2: Create the BigQuery dataset if it does not already exist.
    create_dataset_if_not_exists = BigQueryCreateEmptyDatasetOperator(
        task_id="create_dataset_if_not_exists",
        dataset_id=BIGQUERY_DATASET,
        location=BIGQUERY_LOCATION,
        gcp_conn_id=GCS_CONN_ID, # BQ operators use the default connection
        exists_ok=True,  # This makes the operator idempotent
    )
##axlearn-arc-testing/testing/judywu-poc/unit-tests-cpu-54c49ab-2025-07-23-15_37_57.csv
    # Task 3: Ensure the final destination table exists with partitioning and clustering.
    create_final_table_if_not_exists = BigQueryCreateEmptyTableOperator(
        task_id="create_final_table_if_not_exists",
        gcp_conn_id=GCS_CONN_ID, # BQ operators use the default connection
        dataset_id=BIGQUERY_DATASET,
        table_id=BIGQUERY_FINAL_TABLE,
        schema_fields=[
            {"name": "test_id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "processor", "type": "STRING", "mode": "NULLABLE"},
            {"name": "commit_hash", "type": "STRING", "mode": "NULLABLE"},
            {"name": "run_timestamp", "type": "TIMESTAMP", "mode": "NULLABLE"},
            {"name": "test_path", "type": "STRING", "mode": "NULLABLE"},
            {"name": "module", "type": "STRING", "mode": "NULLABLE"},
            {"name": "name", "type": "STRING", "mode": "NULLABLE"},
            {"name": "file", "type": "STRING", "mode": "NULLABLE"},
            {"name": "doc", "type": "STRING", "mode": "NULLABLE"},
            {"name": "markers", "type": "STRING", "mode": "NULLABLE"},
            {"name": "status", "type": "STRING", "mode": "NULLABLE"},
            {"name": "message", "type": "STRING", "mode": "NULLABLE"},
            {"name": "duration", "type": "FLOAT64", "mode": "NULLABLE"},
        ],
        time_partitioning={"type": "DAY", "field": "run_timestamp"},
        cluster_fields=["processor", "commit_hash"],
    )

    # # check_for_new_files_with_prefix >> move_zipped_files_to_archive >> create_dataset_if_not_exists >> create_final_table_if_not_exists

    # # Task 4: Create a temporary external table pointing to the specific file found.
    create_temp_external_table = BigQueryCreateExternalTableOperator(
        task_id="create_temp_external_table",
        gcp_conn_id=GCS_CONN_ID,
        bucket=GCS_BUCKET,
        source_objects=list_csv_files.output,
        destination_project_dataset_table=f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{TEMP_EXTERNAL_TABLE}",
        schema_fields=[
            {"name": "id", "type": "STRING"},
            {"name": "module", "type": "STRING"},
            {"name": "name", "type": "STRING"},
            {"name": "file", "type": "STRING"},
            {"name": "doc", "type": "STRING"},
            {"name": "markers", "type": "STRING"},
            {"name": "status", "type": "STRING"},
            {"name": "message", "type": "STRING"},
            {"name": "duration", "type": "FLOAT64"},
        ],
        source_format="CSV",
        skip_leading_rows=1,
        max_bad_records=100000,
        allow_jagged_rows=True,
        allow_quoted_newlines=True,
        location="US",
    )

    # # Task 5: Execute the BigQuery job to transform and insert the data.
    transform_and_load = BigQueryInsertJobOperator(
        task_id="transform_and_load_to_bigquery",
        gcp_conn_id=GCS_CONN_ID,
        configuration={
            "query": {
                "query": f"""
                  INSERT INTO `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_FINAL_TABLE}` (
                      test_id, processor, commit_hash, run_timestamp,
                      test_path, module, name, file, doc, markers,
                      status, message, duration
                  )
                  SELECT
                      GENERATE_UUID() AS test_id,
                      COALESCE(REGEXP_EXTRACT(SPLIT(_FILE_NAME, '/')[SAFE_ORDINAL(ARRAY_LENGTH(SPLIT(_FILE_NAME, '/')))], r'unit-tests-([a-z]{{3}})-'), 'unknown') AS processor,
                      COALESCE(REGEXP_EXTRACT(SPLIT(_FILE_NAME, '/')[SAFE_ORDINAL(ARRAY_LENGTH(SPLIT(_FILE_NAME, '/')))], r'unit-tests-[a-z]{{3}}-([a-zA-Z0-9]{{7}})-'), 'unknown_commit') AS commit_hash,
                      PARSE_TIMESTAMP('%Y-%m-%d-%H:%M:%S', REPLACE(REGEXP_EXTRACT(SPLIT(_FILE_NAME, '/')[SAFE_ORDINAL(ARRAY_LENGTH(SPLIT(_FILE_NAME, '/')))], r'(\\d{{4}}-\\d{{2}}-\\d{{2}}-\\d{{2}}[_:]\\d{{2}}[_:]\\d{{2}})'), '_', ':')) AS run_timestamp,
                      csv.id AS test_path,
                      csv.module,
                      csv.name,
                      csv.file,
                      csv.doc,
                      csv.markers,
                      csv.status,
                      csv.message,
                      csv.duration
                  FROM
                `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{TEMP_EXTERNAL_TABLE}` AS csv;
                """,
                "useLegacySql": False,
            }
        },
    )

    # Task 6: Clean up by deleting the temporary external table.
    delete_temp_table = BigQueryDeleteTableOperator(
        task_id="delete_temp_external_table",
        gcp_conn_id=GCS_CONN_ID,
        deletion_dataset_table=f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{TEMP_EXTERNAL_TABLE}",
    )

    # Task 7: Delete all processed csv files
    delete_processed_csv_files = GCSDeleteObjectsOperator(
        task_id="delete_processed_csv_files",
        bucket_name=GCS_BUCKET,
        objects=list_csv_files.output,
        gcp_conn_id=GCS_CONN_ID,
    )

    # --- Define Task Dependencies ---
    list_csv_files >> check_task >> create_dataset_if_not_exists >> create_final_table_if_not_exists
    list_csv_files >> check_task>> create_temp_external_table

    [create_final_table_if_not_exists, create_temp_external_table] >> transform_and_load

    transform_and_load >> [delete_temp_table, delete_processed_csv_files]
