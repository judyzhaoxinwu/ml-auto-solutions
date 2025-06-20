# Copyright 2024 Google LLC
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

"""A DAG to run all supported ML models with the latest JAX/FLAX version."""

import datetime
from airflow import models
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters
from dags.sparsity_diffusion_devx.configs import project_bite_config as config
from airflow.utils.task_group import TaskGroup


# Run once a day at 6 pm UTC (11 am PST)
SCHEDULED_TIME = '0 18 * * *' if composer_env.is_prod_env() else None

common = {
    'test_name': 'bite_tpu_unit_test-jax-0-5-3',
    'time_out_in_min': 60,
    'cluster': XpkClusters.TPU_V5E_256_CLUSTER, ##Project.TPU_PROD_ENV_MULTIPOD
    'docker_image': "us-docker.pkg.dev/tpu-prod-env-multipod/bite/axlearn-unit-test-tpu-v5e-0.5.3:latest",
    'num_slices': 1
}

dag_default_args = {
    "retries": 0,
}

with models.DAG(
    dag_id='project_bite_tpu_e2e_unit_test',
    schedule=SCHEDULED_TIME,
    default_args=dag_default_args,
    tags=[
        'sparsity_diffusion_devx',
        'multipod_team',
        'tpu',
        'axlearn',
        'bite',
    ],
    start_date=datetime.datetime(2025, 6, 4),
    catchup=False,
) as dag:

  with TaskGroup(
      group_id='bite_tpu_unit_test', prefix_group_id=False
  ) as bite_unittests:

    config.get_bite_tpu_unit_test_config(
        **common,
    ).run()
