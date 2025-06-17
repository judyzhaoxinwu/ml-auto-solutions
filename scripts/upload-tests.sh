# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Upload tests and utilities to a specificed GCS dags folder

set -e

COMPOSER_ENVIRONMENT="judyzwu-dev"
GCS_DAGS_FOLDER=$1 # e.g., gs://us-central1-judyzwu-dev-xxxx-bucket/dags

# Define an array of specific subfolders (full path relative to current directory)
# that you want to upload.
# This assumes these specific subfolders exist locally.
SUBFOLDERS_TO_INCLUDE=(
  "dags/common"
  "dags/inference"
  "dags/mlcompass"
  "dags/multipod"
  "dags/sparsity_diffusion_devx"
  "xlml/apis"
  "xlml/utils"
  # Add more specific subfolders like "dags/another_specific_subfolder"
)

echo "GCS_DAGS_FOLDER: $GCS_DAGS_FOLDER"
echo "Subfolders to explicitly include and upload: ${SUBFOLDERS_TO_INCLUDE[@]}"

# TODO(ranran): handle tests from Jsonnet
for subfolder_path in "${SUBFOLDERS_TO_INCLUDE[@]}"; do
  if [ -d "$subfolder_path" ]; then # Check if the local subfolder actually exists
    # Extract the top-level part (e.g., "dags" from "dags/common_utils")
    # This determines the immediate parent folder under GCS_DAGS_FOLDER
    TOP_LEVEL_FOLDER=$(echo "$subfolder_path" | cut -d'/' -f1)

    echo "Processing subfolder: $subfolder_path"
    echo "  --> Uploading to: $GCS_DAGS_FOLDER/$TOP_LEVEL_FOLDER/$subfolder_path"
    # rsync the specific subfolder directly to the desired path in GCS
    # This will create e.g., gs://.../dags/dags/common_utils if TOP_LEVEL_FOLDER is 'dags'
    # and GCS_DAGS_FOLDER is 'gs://.../dags'
    gsutil -m rsync -c -d -r "$subfolder_path" "$GCS_DAGS_FOLDER"/"$subfolder_path"
  else
    echo "Warning: Local subfolder '$subfolder_path' not found. Skipping."
  fi
done

echo "Successfully uploaded specified subfolders."
