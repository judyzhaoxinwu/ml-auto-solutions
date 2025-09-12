from __future__ import print_function
from airflow.models.dag import DAG
# FIXED: This is the correct import for Airflow 1.x
from airflow.operators.python import PythonOperator
from google.cloud import storage
from datetime import datetime, timedelta

# The bucket name
GCS_BUCKET = "axlearn-arc-testing"

# FIXED: The object path should NOT include the bucket name
GCS_OBJECT = "testing/judyzwu_poc/training-test-b200_16-39f69bc-0.5.3-16633833074-2025-07-30-21:29:17.csv"

def generate_gcs_signed_url():
    """Generates a v4 signed URL for a GCS object."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_OBJECT)

    # BEST PRACTICE: Pass a timedelta directly for expiration.
    # This generates a URL that is valid for 15 minutes.
    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=15),
        method="GET",
    )

    print(f"Generated GCS Signed URL: {signed_url}")
    return signed_url


with DAG(
    dag_id="gcs_signed_url_generation",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["gcs", "signed_url"],
) as dag:
    generate_url_task = PythonOperator(
        task_id="generate_signed_url",
        python_callable=generate_gcs_signed_url,
    )
