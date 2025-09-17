"""
DAG to process unit test result CSV files from GCS, enrich the data with GCS
metadata, and load it into a final BigQuery table.

This DAG uses a robust ELT pattern with dynamic task mapping:
1.  Lists all new files in a GCS bucket.
2.  For each file found, it dynamically spawns a series of tasks:
    a. Fetches the GCS object's custom metadata.
    b. Creates a temporary external table pointing to the file.
    c. Runs a transform-and-load query to enrich data using the fetched
       metadata and inserts it into the final table.
    d. Cleans up by deleting the temporary table and the source file.
"""
from __future__ import annotations

import datetime
import re

from airflow.models.dag import DAG
from airflow.decorators import task
from airflow.providers.google.cloud.hooks.gcs import GCSHook

from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCreateEmptyDatasetOperator,
    BigQueryCreateEmptyTableOperator,
    BigQueryCreateExternalTableOperator,
    BigQueryDeleteTableOperator,
    BigQueryInsertJobOperator,
)
from airflow.providers.google.cloud.operators.gcs import (
    GCSDeleteObjectsOperator,
    GCSListObjectsOperator,
)


# --- Configuration Variables ---
GCP_PROJECT_ID = "tpu-prod-env-one-vm"
GCS_BUCKET = "axlearn-arc-testing"
GCS_SOURCE_FOLDER = "testing/results/"  ##change here for testing
BIGQUERY_DATASET = "axlearn_arc_testing"
BIGQUERY_FINAL_TABLE = "axlearn_test_results"  ##change here for testing
GCS_CONN_ID = "gcs_external_project_conn"
BIGQUERY_LOCATION = "US"
GITHUB_RUN_LINK_PREFIX = "https://github.com/Borklet-Labs/axlearn-arc/actions/runs/"

# This variable is no longer used by the mapped tasks
# TEMP_EXTERNAL_TABLE = "temp_external_test_runs_{{ ts_nodash }}"

with DAG(
    dag_id="gcs_metadata_to_bigquery_dynamic",
    start_date=datetime.datetime(2025, 7, 21),
    schedule="0 */8 * * *",
    catchup=False,
    tags=["bigquery", "gcs", "testing", "dynamic-tasks"],
    render_template_as_native_obj=True,  # Important for passing lists
) as dag:
    list_csv_files = GCSListObjectsOperator(
        task_id="list_csv_files",
        bucket=GCS_BUCKET,
        prefix=GCS_SOURCE_FOLDER,
        match_glob=f"{GCS_SOURCE_FOLDER}*.csv",
        gcp_conn_id=GCS_CONN_ID,
    )

    @task.short_circuit
    def check_if_files_exist(files_found: list) -> bool:
        print(f"Files found: {files_found}")
        return bool(files_found)

    check_task = check_if_files_exist(files_found=list_csv_files.output)

    create_dataset_if_not_exists = BigQueryCreateEmptyDatasetOperator(
        task_id="create_dataset_if_not_exists",
        dataset_id=BIGQUERY_DATASET,
        location=BIGQUERY_LOCATION,
        gcp_conn_id=GCS_CONN_ID,
        exists_ok=True,
    )

    create_final_table_if_not_exists = BigQueryCreateEmptyTableOperator(
        task_id="create_final_table_if_not_exists",
        gcp_conn_id=GCS_CONN_ID,
        dataset_id=BIGQUERY_DATASET,
        table_id=BIGQUERY_FINAL_TABLE,
        schema_fields=[
            {"name": "test_id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "test_type", "type": "STRING", "mode": "NULLABLE"},
            {"name": "processor", "type": "STRING", "mode": "NULLABLE"},
            {"name": "accelerator", "type": "STRING", "mode": "NULLABLE"},
            {"name": "jax_version", "type": "STRING", "mode": "NULLABLE"},
            {"name": "github_run_id", "type": "STRING", "mode": "NULLABLE"},
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

    # --- YOUR MODIFIED (AND CORRECT) METADATA TASK ---
    @task
    def get_gcs_metadata_with_fallback(gcs_object_paths: list, gcs_bucket: str):
        """
        Fetches custom metadata from a GCS object. If key metadata fields are
        missing, it falls back to parsing the filename.
        """
        hook = GCSHook(gcp_conn_id=GCS_CONN_ID)
        all_metadata = []
        for path in gcs_object_paths:
            metadata = hook.get_metadata(bucket_name=gcs_bucket, object_name=path)

            print(f"Metadata for {path}: {metadata}")
            now_dt = datetime.datetime.now()
            default_timestamp_str = now_dt.strftime("%Y-%m-%d-%H:%M:%S")
            # Check if the new metadata exists and is populated
            if metadata is not None:
                print(f"Found new metadata for: {path}")
                result = {
                    "source_logic": "metadata",
                    "test_type": metadata.get("test-type", ""),
                    "processor": metadata.get("processor", ""),
                    "accelerator": metadata.get("accelerator", ""),
                    "jax_version": metadata.get("jax-version", ""),
                    "github_run_id": metadata.get("github-run-id", ""),
                    "commit_hash": metadata.get("commit-hash", ""),
                    "run_timestamp": metadata.get("run-timestamp", default_timestamp_str),
                }
            else:
                # FALLBACK LOGIC: If metadata is missing, parse the filename
                print(f"Metadata not found. Falling back to filename parsing for: {path}")
                filename = path.split("/")[-1]
                print(f"extracing file name: {filename}")

                # Helper function to run regex and get group 1, or a default value
                def extract(pattern, text, default=""):
                    match = re.search(pattern, text)
                    return match.group(1) if match else default

                # Translate the BigQuery regex to Python's re module
                test_type = extract(r"^([a-z]+-tests?)", filename)
                processor = (
                    extract(r"-(cpu|gpu|tpu)-", filename)
                    if test_type == "unit-tests"
                    else ""
                )
                accelerator = (
                    extract(r"test-(.*?)-[a-f0-9]{7}-\d\.\d\.\d-", filename)
                    if test_type == "training-test"
                    else ""
                )
                jax_version = extract(r"-(\d\.\d\.\d(?:\.dev\d+)?)-", filename)
                github_run_id = extract(
                    r"-\d\.\d\.\d(?:\.dev\d+)?-([0-9]+)-", filename
                )
                commit_hash = extract(
                    r"-([a-f0-9]{7})-\d\.\d\.\d(?:\.dev\d+)?-", filename
                )

                # Reconstruct the timestamp from the filename
                ts_match = extract(
                    r"(\d{4}-\d{2}-\d{2}-\d{2}[_:]\d{2}[_:]\d{2})", filename
                )
                formatted_ts = (
                    ts_match.replace("_", ":") if ts_match else default_timestamp_str
                )

                result = {
                    "source_logic": "filename",
                    "test_type": test_type,
                    "processor": processor,
                    "accelerator": accelerator,
                    "jax_version": jax_version,
                    "github_run_id": github_run_id,
                    "commit_hash": commit_hash,
                    "run_timestamp": formatted_ts,
                }
            all_metadata.append(result)
        return all_metadata

    # --- YOUR MODIFIED (AND CORRECT) METADATA TASK CALL ---
    fetched_metadata = get_gcs_metadata_with_fallback(
        gcs_bucket=GCS_BUCKET, gcs_object_paths=list_csv_files.output
    )

    # --- NEW HELPER TASKS (FOR FIX 1) ---
    @task
    def prep_source_objects_list(files: list[str]) -> list[list[str]]:
        """Turns ['a', 'b'] into [['a'], ['b']] for mapping source_objects."""
        return [[f] for f in files]

    @task
    def prep_table_names(files: list[str], ts_nodash: str) -> list[dict]:
        """
        Generates a list of unique table names and IDs for each file,
        using the map index to ensure uniqueness.
        """
        tables = []
        for i, f in enumerate(files):
            # Use the index 'i' to make the table name unique for each mapped task
            short_name = f"temp_external_test_runs_{ts_nodash}_{i}"
            tables.append(
                {
                    "short_name": short_name,
                    "full_id": f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{short_name}",
                }
            )
        return tables

    @task
    def combine_inputs_for_bq(meta_list: list, table_names_list: list[dict]):
        """
        Combines metadata and table names into a single list for the
        build_bq_load_config_task, as it can only map over one input.
        """
        return [
            {"meta": meta, "table_info": table_info}
            for meta, table_info in zip(meta_list, table_names_list)
        ]

    # --- NEW HELPER TASK CALLS (FOR FIX 1) ---
    mapped_source_objects = prep_source_objects_list(list_csv_files.output)

    table_names_list = prep_table_names(
        files=list_csv_files.output, ts_nodash="{{ ts_nodash }}"
    )

    combined_bq_inputs = combine_inputs_for_bq(
        meta_list=fetched_metadata, table_names_list=table_names_list
    )

    # --- REPLACED `create_temp_external_table` (FOR FIX 1) ---
    # This task is now MAPPED to run once per file
    create_temp_external_table = BigQueryCreateExternalTableOperator.partial(
        task_id="create_temp_external_table",
        gcp_conn_id=GCS_CONN_ID,
        bucket=GCS_BUCKET,
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
    ).expand(
        source_objects=mapped_source_objects,
        destination_project_dataset_table=table_names_list.map(lambda x: x["full_id"]),
    )

    # --- REPLACED `build_bq_load_config_task` (FOR FIX 1) ---
    # This task now takes the combined input and uses the UNIQUE table name
    @task
    def build_bq_load_config_task(combined_input: dict) -> dict:
        """
        Takes one combined {meta, table_info} dict
        and returns one complete BQ job configuration dict.
        """
        meta = combined_input["meta"]
        # This is now the unique table name, e.g., temp_external_test_runs_..._0
        temp_table_name = combined_input["table_info"]["short_name"]

        # Construct the query string from the metadata
        insert_query = f"""
        INSERT INTO `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_FINAL_TABLE}` (
            test_id, test_type, processor, accelerator, commit_hash, jax_version, github_run_id,
            run_timestamp, test_path, module, name, file, doc, markers,
            status, message, duration
        )
        SELECT
            GENERATE_UUID() AS test_id,
            '{meta["test_type"]}' AS test_type,
            '{meta["processor"]}' AS processor,
            '{meta["accelerator"]}' AS accelerator,
            '{meta["commit_hash"]}' AS commit_hash,
            '{meta["jax_version"]}' AS jax_version,
            CONCAT('{GITHUB_RUN_LINK_PREFIX}', '{meta["github_run_id"]}') AS github_run_id,
            PARSE_TIMESTAMP('%Y-%m-%d-%H:%M:%S', '{meta["run_timestamp"]}') AS run_timestamp,
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
        -- This now uses the UNIQUE temp table name
        `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{temp_table_name}` AS csv;
        """

        # Return the complete configuration dictionary
        return {
            "query": {
                "query": insert_query,
                "useLegacySql": False,
            }
        }

    # --- REPLACED `list_of_bq_configs` CALL (FOR FIX 1) ---
    # This now maps over the combined input
    list_of_bq_configs = build_bq_load_config_task.expand(
        combined_input=combined_bq_inputs
    )

    # This task is correct and maps over the list of configs
    transform_and_load = BigQueryInsertJobOperator.partial(
        task_id="transform_and_load_to_bigquery",
        gcp_conn_id=GCS_CONN_ID,
    ).expand(configuration=list_of_bq_configs)

    # --- REPLACED `delete_temp_table` (FOR FIX 1) ---
    # This task is now MAPPED to delete all the unique temp tables
    delete_temp_table = BigQueryDeleteTableOperator.partial(
        task_id="delete_temp_external_table", gcp_conn_id=GCS_CONN_ID
    ).expand(
        deletion_dataset_table=table_names_list.map(lambda x: x["full_id"])
    )

    delete_processed_csv_files = GCSDeleteObjectsOperator(
        task_id="delete_processed_csv_files",
        bucket_name=GCS_BUCKET,
        objects=list_csv_files.output,
        gcp_conn_id=GCS_CONN_ID,
    )
    # --- REPLACED DEPENDENCIES (FOR FIX 1) ---
    check_task >> [create_dataset_if_not_exists, create_final_table_if_not_exists]

    # These all branch from check_task and are based on the file list
    check_task >> fetched_metadata
    check_task >> mapped_source_objects
    check_task >> table_names_list

    # The BQ config builder needs both metadata and table names
    [fetched_metadata, table_names_list] >> combined_bq_inputs

    # The external table creation needs the source objects and table names
    [mapped_source_objects, table_names_list] >> create_temp_external_table

    # The BQ config builder must wait for the temp tables to be created
    # and the final table to exist.
    [
        create_final_table_if_not_exists,
        create_temp_external_table,
        combined_bq_inputs,
    ] >> list_of_bq_configs

    list_of_bq_configs >> transform_and_load

    # Clean up the temp tables
    transform_and_load >> delete_temp_table >> delete_processed_csv_files
