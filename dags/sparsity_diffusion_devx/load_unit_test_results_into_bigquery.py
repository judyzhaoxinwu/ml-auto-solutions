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
from operator import itemgetter

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
BIGQUERY_FINAL_TABLE = "axlearn_test_results"  ##change here for testing + _jwu_test
GCS_CONN_ID = "gcs_external_project_conn"
BIGQUERY_LOCATION = "US"
GITHUB_RUN_LINK_PREFIX = "https://github.com/Borklet-Labs/axlearn-arc/actions/runs/"


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
                    "file_name": path,
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

                # --- CORRECTED REGEX LOGIC ---

                # Pattern to CAPTURE the JAX version (has outer parens)
                JAX_VERSION_CAPTURE_PATTERN = r"(\d\.\d\.\d(?:\.dev\d+)?)"
                # Pattern to MATCH the JAX version (no outer parens, for anchoring)
                JAX_VERSION_MATCH_PATTERN = r"\d\.\d\.\d(?:\.dev\d+)?"

                # CORRECTED: This regex now captures only the exact string you want
                test_type = extract(r"^(unit-tests|training-test)", filename)

                # This logic is still correct and will work with the new test_type
                processor = (
                    extract(r"-(cpu|gpu|tpu)-", filename)
                    if "unit-tests" in test_type
                    else ""
                )

                # This logic is also still correct
                accelerator = (
                    extract(r"^training-test-(.*?)-[a-f0-9]{7}-", filename)
                    if "training-test" in test_type
                    else ""
                )

                # CORRECTED: Uses the CAPTURE pattern
                jax_version = extract(r"-" + JAX_VERSION_CAPTURE_PATTERN + r"-", filename)

                # CORRECTED: Uses the MATCH pattern for anchoring
                # This ensures "([0-9]+)" is now Group 1
                github_run_id = extract(
                    r"-" + JAX_VERSION_MATCH_PATTERN + r"-([0-9]+)-", filename
                )

                # CORRECTED: Uses the MATCH pattern for anchoring for robustness
                commit_hash = extract(
                    r"-([a-f0-9]{7})-" + JAX_VERSION_MATCH_PATTERN + r"-", filename
                )

                # --- END OF CORRECTIONS ---

                # Reconstruct the timestamp from the filename
                ts_match = extract(
                    r"(\d{4}-\d{2}-\d{2}-\d{2}[_:]\d{2}[_:]\d{2})", filename
                )
                formatted_ts = (
                    ts_match.replace("_", ":") if ts_match else default_timestamp_str
                )

                result = {
                    "source_logic": "filename",
                    "file_name": path,
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

    fetched_metadata = get_gcs_metadata_with_fallback(
        gcs_bucket=GCS_BUCKET, gcs_object_paths=list_csv_files.output
    )

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
                    "file_name": f,
                }
            )
        return tables

    @task
    def combine_inputs_for_bq(meta_list: list, table_names_list: list[dict]):
        """
        Combines metadata and table names into a single list, filters to ensure
        file names match, and then sorts the list by timestamp for reliable mapping.
        """
        combined_list = []

        # 1. Combine and Filter (Explicit Coupling Check)
        for meta, table_info in zip(meta_list, table_names_list):
            # Assuming you've already added file_name to both meta and table_info in upstream tasks
            if meta.get("file_name") == table_info.get("file_name"):
                combined_list.append({
                    "meta": meta,
                    "table_info": table_info
                })
            else:
                # Log a warning if a mismatch occurs
                print(f"WARNING: File mismatch found. Skipping: {meta.get('file_name')} != {table_info.get('file_name')}")

        sorted_combined_list = sorted(
            combined_list,
            key=lambda item: item["meta"]["run_timestamp"]
        )
        file_names = [x["table_info"]["file_name"] for x in sorted_combined_list]
        full_ids = [x["table_info"]["full_id"] for x in sorted_combined_list]

        # Print the resulting lists
        print(f"Sorted combined list file names: {file_names}")
        print(f"Sorted combined list full ids: {full_ids}")
        return sorted_combined_list


    mapped_source_objects = prep_source_objects_list(list_csv_files.output)

    table_names_list = prep_table_names(
        files=list_csv_files.output, ts_nodash="{{ ts_nodash }}"
    )

    combined_bq_inputs = combine_inputs_for_bq(
        meta_list=fetched_metadata, table_names_list=table_names_list
    )

    @task
    def prep_mapped_bq_inputs(combined_inputs: list) -> list[dict]:
        """
        Takes the combined list and extracts only the specific arguments
        needed for the BigQueryCreateExternalTableOperator.
        """
        return [
            {
                "source_objects": [x["table_info"]["file_name"]], # CRITICAL: Re-wrap the file name in a list
                "destination_project_dataset_table": x["table_info"]["full_id"],
            }
            for x in combined_inputs
        ]


    bq_mapped_configs = prep_mapped_bq_inputs(combined_inputs=combined_bq_inputs)

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
    ).expand_kwargs(
        bq_mapped_configs
    )

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

    list_of_bq_configs = build_bq_load_config_task.expand(
        combined_input=combined_bq_inputs
    )

    transform_and_load = BigQueryInsertJobOperator.partial(
        task_id="transform_and_load_to_bigquery",
        gcp_conn_id=GCS_CONN_ID,
    ).expand(configuration=list_of_bq_configs)


    delete_temp_table = BigQueryDeleteTableOperator.partial(
        task_id="delete_temp_external_table", gcp_conn_id=GCS_CONN_ID
    ).expand(
        deletion_dataset_table=table_names_list.map(lambda x: x["full_id"])
    )

    @task(max_active_tis_per_dag=32) # Set a high limit to allow parallel deletions
    def delete_gcs_file(file_path: str, bucket_name: str, gcp_conn_id: str):
        """Deletes a single GCS object using the GCSHook for reliability."""
        if not file_path:
            print("WARNING: Received an empty file path. Skipping deletion.")
            return

        # Initialize hook inside the task for reliability
        hook = GCSHook(gcp_conn_id=gcp_conn_id)
        print(f"Attempting to delete GCS object: {bucket_name}/{file_path}")

        try:
            # Use the hook's delete method to remove the single object
            hook.delete(bucket_name=bucket_name, object_name=file_path)
            print(f"Successfully deleted: {file_path}")
        except Exception as e:
            # Catch NotFound errors specifically, in case a file was already deleted
            if "No such object" in str(e):
                print(f"Warning: File {file_path} not found (already deleted or truncated). Continuing.")
            else:
                # Re-raise other errors
                raise e

    # The list of file paths to delete is extracted from combined_bq_inputs
    files_to_delete_list = combined_bq_inputs.map(lambda x: x["table_info"]["file_name"])

    # The deletion task is mapped over the clean list of paths
    delete_processed_csv_files = delete_gcs_file.partial(
        bucket_name=GCS_BUCKET,
        gcp_conn_id=GCS_CONN_ID,
    ).expand(
        file_path=files_to_delete_list,
    )

    check_task >> [create_dataset_if_not_exists, create_final_table_if_not_exists]
    check_task >> [fetched_metadata, table_names_list]

    [fetched_metadata, table_names_list] >> combined_bq_inputs

    combined_bq_inputs >> [bq_mapped_configs, list_of_bq_configs]

    bq_mapped_configs >> create_temp_external_table

    [create_final_table_if_not_exists, create_temp_external_table] >> transform_and_load

    transform_and_load >> delete_temp_table

    transform_and_load >> delete_processed_csv_files
