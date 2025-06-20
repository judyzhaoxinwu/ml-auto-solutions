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
from dags.common.vm_resource import MachineVersion, ImageFamily, GpuVersion, Zone, RuntimeVersion
from dags.sparsity_diffusion_devx.configs import project_bite_config as config
from airflow.utils.task_group import TaskGroup
from dags.common.vm_resource import XpkClusters, DockerImage
from datetime import timedelta


# Run once a day at 6 pm UTC (11 am PST)
SCHEDULED_TIME = '0 18 * * *' if composer_env.is_prod_env() else None

# common = {
#     'time_out_in_min': 300,
#     'test_name': "bite_gpu_unittest-jax-0-5-3",
#     'cluster': XpkClusters.GPU_A3PLUS_CLUSTER,
#     'docker_image': "us-docker.pkg.dev/tpu-prod-env-multipod/bite/axlearn-unit-test-gpu-a3-h100-0.5.3:latest",
#     'num_slices': 1,
# }

common = {
    "machine_type": MachineVersion.A2_HIGHGPU_4G.value,
    "image_family": ImageFamily.COMMON_CU124_DEBIAN_11.value, ##"tf-ent-latest-gpu",
    "count": 4,
    "gpu_zone": Zone.US_WEST1_B.value,
    "accelerator_type": GpuVersion.A100.value,
    "runtime_version": RuntimeVersion.TPU_UBUNTU2204_BASE.value,
    "network": "projects/tpu-prod-env-multipod/global/networks/mas-test",
    "subnetwork": "projects/tpu-prod-env-multipod/global/networks/xlml/regions/us-west4/subnetworks/mas-test",
    "jax_version": "0.5.3",
    "project_name": "tpu-prod-env-multipod",
}

dag_default_args = {
    "retries": 0,
}

with models.DAG(
    dag_id='project_bite_gpu_e2e_unit_test',
    schedule=SCHEDULED_TIME,
    default_args=dag_default_args,
    tags=[
        'sparsity_diffusion_devx',
        'multipod_team',
        'gpu',
        'axlearn',
        'bite',
    ],
    start_date=datetime.datetime(2025, 6, 4),
    catchup=False,
) as dag:
  with TaskGroup(
      group_id='bite_gpu_unittests', prefix_group_id=False
  ) as bite_unittests:
    config.get_bite_gpu_unittests_config(
        **common,
    ).run()
