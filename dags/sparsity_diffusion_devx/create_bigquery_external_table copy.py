#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
### Airflow DAG for GCS to BigQuery with Auto-Generated Unique ID (Append Pattern)

This DAG demonstrates an ETL pattern for ingesting data from GCS into a pre-existing
BigQuery table, ensuring each new row has a unique identifier.

**Pattern:**
1.  **Create Final Table:** An empty native BigQuery table is created with the full
    final schema, including a column for the unique ID. This step is idempotent
    and will not fail if the table already exists.
2.  **Create External Table:** A temporary external table is created in BigQuery that
    points directly to the CSV files in the specified GCS bucket.
3.  **Insert and Transform Data:** An `INSERT INTO ... SELECT` query is run. This query
    selects data from the external table, generates a unique ID, and inserts the
    transformed records into the empty native table created in step 1.
4.  **Cleanup:** The temporary external table is deleted.
"""

from __future__ import annotations

from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCreateEmptyTableOperator,
    BigQueryCreateExternalTableOperator,
    BigQueryDeleteTableOperator,
    BigQueryInsertJobOperator,
)
from airflow.providers.google.cloud.transfers.gcs_to_gcs import GCSToGCSOperator
from airflow.utils.dates import days_ago

# Define your project, dataset, and table details
PROJECT_ID = "cloud-ml-auto-solutions"
DATASET_ID = "xlml_bite_testresults"
# Final, permanent table that will store the data with a unique ID
FINAL_TABLE_ID = "axlearn_unit_test_results"
# Temporary external table to read from GCS
TEMP_TABLE_ID = "axlearn_unit_test_results_temp"

# GCS location of the source files
GCS_BUCKET = "ml-auto-solutions"
GCS_SOURCE_PATH = "output/sparsity_diffusion_devx/axlearn-unit-tests/test-results"
GCS_ARCHIVE_PATH = "output/sparsity_diffusion_devx/axlearn-unit-tests/archive"


# The schema of the source CSV files in GCS. This matches the raw data.
SOURCE_TABLE_SCHEMA = [
    {"name": "id", "type": "STRING", "mode": "NULLABLE"},
    {"name": "module", "type": "STRING", "mode": "NULLABLE"},
    {"name": "name", "type": "STRING", "mode": "NULLABLE"},
    {"name": "file", "type": "STRING", "mode": "NULLABLE"},
    {"name": "doc", "type": "STRING", "mode": "NULLABLE"},
    {"name": "markers", "type": "STRING", "mode": "NULLABLE"},
    {"name": "status", "type": "STRING", "mode": "NULLABLE"},
    {"name": "message", "type": "STRING", "mode": "NULLABLE"},
    {"name": "duration", "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "platform", "type": "STRING", "mode": "NULLABLE"},
    {"name": "accelerator_type", "type": "STRING", "mode": "NULLABLE"},
    {"name": "datetime", "type": "TIMESTAMP", "mode": "NULLABLE"},
    {"name": "test_name", "type": "STRING", "mode": "NULLABLE"},
    {"name": "jax_version", "type": "STRING", "mode": "NULLABLE"},
]

# The schema for our final, destination table in BigQuery.
# It includes the new 'unique_id' column.
FINAL_TABLE_SCHEMA = [
    {"name": "unique_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "id", "type": "STRING", "mode": "NULLABLE"},
    {"name": "module", "type": "STRING", "mode": "NULLABLE"},
    {"name": "name", "type": "STRING", "mode": "NULLABLE"},
    {"name": "file", "type": "STRING", "mode": "NULLABLE"},
    {"name": "doc", "type": "STRING", "mode": "NULLABLE"},
    {"name": "markers", "type": "STRING", "mode": "NULLABLE"},
    {"name": "status", "type": "STRING", "mode": "NULLABLE"},
    {"name": "message", "type": "STRING", "mode": "NULLABLE"},
    {"name": "duration", "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "platform", "type": "STRING", "mode": "NULLABLE"},
    {"name": "accelerator_type", "type": "STRING", "mode": "NULLABLE"},
    {"name": "datetime", "type": "TIMESTAMP", "mode": "NULLABLE"},
    {"name": "test_name", "type": "STRING", "mode": "NULLABLE"},
    {"name": "jax_version", "type": "STRING", "mode": "NULLABLE"},
]


# SQL query to insert data from the external table into the final table.
INSERT_INTO_FINAL_TABLE_SQL = f"""
INSERT INTO `{PROJECT_ID}.{DATASET_ID}.{FINAL_TABLE_ID}`
SELECT
    -- Generate a unique identifier for each row
    GENERATE_UUID() AS unique_id,
    -- Select and cast all other columns from the source
    id,
    module,
    name,
    file,
    doc,
    markers,
    status,
    message,
    CAST(duration AS FLOAT64) AS duration,
    platform,
    accelerator_type,
    -- Cast the string datetime from CSV to a proper TIMESTAMP
    CAST(datetime AS TIMESTAMP) AS datetime,
    test_name,
    jax_version
FROM
    `{PROJECT_ID}.{DATASET_ID}.{TEMP_TABLE_ID}`;
"""


with DAG(
    dag_id="gcs_to_bigquery_with_uuid_append_dag",
    start_date=days_ago(1),
    schedule=None,
    catchup=False,
    tags=["bigquery", "gcs", "etl"],
    default_args={"gcp_conn_id": "google_cloud_default", "location": "us-central1"},
) as dag:
    # Task 1: Create an empty table to store the final results.
    # This task won't fail if the table already exists.
    create_final_table = BigQueryCreateEmptyTableOperator(
        task_id="create_final_empty_table",
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=FINAL_TABLE_ID,
        schema_fields=FINAL_TABLE_SCHEMA,
        exists_ok=True,
        location="US",
    )

    # Task 2: Create a temporary BigQuery external table linked to the GCS files.
    create_external_table = BigQueryCreateExternalTableOperator(
        task_id="create_external_table",
        bucket=GCS_BUCKET,
        source_objects=[f"{GCS_SOURCE_PATH}/*.csv"],
        destination_project_dataset_table=f"{PROJECT_ID}.{DATASET_ID}.{TEMP_TABLE_ID}",
        schema_fields=SOURCE_TABLE_SCHEMA,
        source_format="CSV",
        skip_leading_rows=1,  # Assuming the CSV files have a header row
        # Parameters for error tolerance and debugging CSV files.
        max_bad_records=100000,
        allow_jagged_rows=True,
        allow_quoted_newlines=True,
        location="US",
    )

    # Task 3: Insert data from the external table into the final native table.
    insert_data_with_uuid = BigQueryInsertJobOperator(
        task_id="insert_data_with_uuid",
        configuration={
            "query": {
                "query": INSERT_INTO_FINAL_TABLE_SQL,
                "useLegacySql": False,
                # For idempotent DAG runs, you might want to TRUNCATE before inserting.
                # "writeDisposition": "WRITE_TRUNCATE",
            }
        },
        location="US",
    )

    # Task 4: Move the processed files to an archive folder to prevent reprocessing.
    move_processed_files_to_archive = GCSToGCSOperator(
        task_id="move_processed_files_to_archive",
        source_bucket=GCS_BUCKET,
        source_object=f"{GCS_SOURCE_PATH}/*.csv",
        destination_bucket=GCS_BUCKET,
        destination_object=f"{GCS_ARCHIVE_PATH}/",
        move_object=True,
    )

    # Task 5: Delete the temporary external table.
    delete_external_table = BigQueryDeleteTableOperator(
        task_id="delete_external_table",
        deletion_dataset_table=f"{PROJECT_ID}.{DATASET_ID}.{TEMP_TABLE_ID}",
        ignore_if_missing=True,
    )

    # Define the task dependency chain.
    # The final table and external table can be created in parallel.
    (
        [create_final_table, create_external_table]
        >> insert_data_with_uuid
        >> move_processed_files_to_archive
        >> delete_external_table
    )
