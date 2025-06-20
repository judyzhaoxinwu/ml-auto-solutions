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

"""Utilities to construct configs for solutionsteam_jax_bite DAG."""


import datetime
from typing import Optional
from dags.common import test_owner
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import gcs_bucket
from dags.sparsity_diffusion_devx.configs import cmd_config
from dags.common.vm_resource import TpuVersion, Project
# from airflow.models.taskmixin import DAGNode
from dags.common.vm_resource import ImageProject, Project, Project, XpkClusters
from xlml.apis.xpk_cluster_config import XpkClusterConfig


GCS_SUBFOLDER_PREFIX = test_owner.Team.SPARSITY_DIFFUSION_DEVX.value

def get_bite_tpu_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    runtime_version: str,
    model_config: str,
    time_out_in_min: int,
    task_owner: str,
    is_tpu_reserved: bool = False,
    jax_version: Optional[str] = None,
    pinned_version: Optional[str] = None,
    project_name: Optional[Project] = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    network: str = "default",
    subnetwork: str = "default",
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = cmd_config.set_up_axlearn(pinned_version, jax_version)
  run_model_cmds = (
      (
          "cd axlearn && python -m axlearn.common.launch_trainer_main"
          f" --module=text.gpt.c4_trainer --config={model_config}"
          f" --trainer_dir={metric_config.SshEnvVars.GCS_OUTPUT.value}"
          f" --data_dir={gcs_bucket.AXLEARN_DIR} --jax_backend=tpu"
      ),
  )

  test_name = f"bite_tpu_training_{'pinned_' if pinned_version else ''}{model_config}_{jax_version.replace('.', '-') if jax_version else 'main'}"
  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=is_tpu_reserved,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=task_owner,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/jax",
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )

##for TPU unit test with pre-built docker images
def get_bite_tpu_unit_test_config(
    test_name: str,
    cluster: XpkClusterConfig,
    docker_image: str,
    time_out_in_min: int,
    num_slices: int=1,
)-> task.XpkTask:
  # Run the unittest as non-root user, ulimit param req to mmap TPUs inside docker (default limit is 8192)
  unittest_runcmds = (
      'echo "#### Start docker image - cpu_unittests"',
      "mkdir -p test-results",
      "export JAX_ENABLE_X64=True",
      "echo gs://ml-auto-solutions/output/sparsity_diffusion_devx/axlearn-unit-tests",
      'pytest --no-header -v -m "tpu or for_8_devices" --dist worksteal '
      "--csv=test-results/bite_axlearn_unit_test_tpu_jax_0_5_3_results.csv "
      "--csv-columns id,module,name,file,doc,markers,status,message,duration,platform,accelerator_type,datetime,test_name,jax_version "
      "--ignore axlearn/common/inference_test.py "
      "--ignore axlearn/common/flash_attention/utils_test.pym "
      "--ignore axlearn/common/flash_attention/neuron_attention_test.py || true && "
      "TESTS_EXIT_CODE=\\$? && "
      'echo "#### TPU JAX Tests finished." && '
      'echo "Test exit code is \\${TESTS_EXIT_CODE}" && '
      'echo "\\${TESTS_EXIT_CODE}" > /workspace/axlearn/test-results/tests_exit_code.txt && '
      "cp -av /workspace/axlearn/test-results /tmp_docker/ && "
      "gcloud storage cp -R test-results/*.csv gs://ml-auto-solutions/output/sparsity_diffusion_devx/axlearn-unit-tests/test-results/ && "
      'echo "Tests exit code: \\$(cat test-results/tests_exit_code.txt)" && '
      "if [[ `cat test-results/tests_exit_code.txt` -ne 0 ]]; then exit 1; fi"
  )

  job_gcp_config = gcp_config.GCPConfig(
      project_name=cluster.project,
      zone=cluster.zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  tpu_unit_test_config = test_config.TpuGkeTest(
      test_config.Tpu(
          version=cluster.device_version,
          cores=cluster.core_count,
      ),
      test_name=test_name,
      cluster_name=cluster.name,
      docker_image=docker_image,
      set_up_cmds=None,
      run_model_cmds=unittest_runcmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.Judy_W,
      num_slices=num_slices,
  )
  return task.XpkTask(
      task_test_config=tpu_unit_test_config,
      task_gcp_config=job_gcp_config,
  )

##for the previous TPU unit test confi  g with dynamic built Docker image
def get_bite_tpu_unittests_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    runtime_version: str,
    time_out_in_min: int,
    task_owner: str,
    network: str = "default",
    subnetwork: str = "default",
    is_tpu_reserved: bool = False,
    # JAX version defaults to main if not specified
    jax_version: Optional[str] = None,
    project_name: Optional[Project] = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
):
  """Function to create Cloud Composer tasks required to run TPU unittests.

  NB! If any of the pytest tests in Axlearn fail, then the whole task will be
  marked as failed.

  1. Creates list of setup commands which will:
    - Create a dockerfile configured to install Axlearn from github, including
      required prerequists and install the desired version of JAX
    - Create a test script to run inside that docker image (run_tpu_tests.sh)
      which runs pytest in the axlearn directory
    - Build a docker image - dockerfile_build_cmd() from dockerfile
  2. Creates list of test run commands which will:
    - On the TPU VM, run the docker image and execute the bash test script
    - Save the STDOUT to a log file, Save the docker logs to a file and copy these
      logs and the Axlearn XML test result output to the GCS output bucket
    - Get the exit code of the pytest command (in the container) and pass it back
      to the TPU VM code which then returns failure for the Airflow task.
  3. Uses test_config.TpuVmTest() class to generate a TPU VM/SSH keys, etc
    and run the setup commands (set_up_cmds) and then the test commands
    (run_model_cmds)

  Returns:
      A task group generated from the TpuVmTest() class.
  """

  pytest_cmds=(f"""pytest --no-header -v -m "tpu or for_8_devices" --dist worksteal \
  --csv=test-results/bite_axlearn_unit_test_tpu_jax_{jax_version}_results.csv \
  --csv-columns id,module,name,file,doc,markers,status,message,duration,platform,accelerator_type,datetime,test_name,jax_version \
  --ignore axlearn/common/inference_test.py \
  --ignore axlearn/common/flash_attention/utils_test.py \
  --ignore axlearn/common/flash_attention/neuron_attention_test.py""")

  unittest_setupcmds = (
      # create configuration files needed
      cmd_config.dockerfile_build_cmd(jax_version),
      cmd_config.pytest_env_setup_cmd(platform="tpu", jax_version=jax_version, accelerator_type="v5p", pytest_cmds=pytest_cmds),
      "chmod +x run_tpu_tests.sh",
      "cat Dockerfile_CI",
      "cat run_tpu_tests.sh",
      "sudo docker build -f Dockerfile_CI -t ml-auto-solutions/tpu_unittests .",
  )

  # Run the unittest as non-root user, ulimit param req to mmap TPUs inside docker (default limit is 8192)
  unittest_runcmds = (
      "echo '#### Start docker image - tpu_unittests'",
      "mkdir -p test-results",
      "sudo chown -R $(whoami):$(whoami) test-results",
      "sudo docker run --shm-size='8g' --network=host --privileged --ulimit memlock=-1:-1 -v ${PWD}:/tmp_docker ml-auto-solutions/tpu_unittests  /bin/bash -c '/workspace/run_tpu_tests.sh' 2>&1 | tee test-results/tests_std_out_err.log",
      "sudo docker logs $( sudo docker ps --latest --quiet ) > test-results/docker_log.log",
      f"echo {metric_config.SshEnvVars.GCS_OUTPUT.value}axlearn-test-results",
      f"gcloud storage cp -R test-results {metric_config.SshEnvVars.GCS_OUTPUT.value}axlearn-test-results",
      "echo 'Tests exit code: '$(cat test-results/tests_exit_code.txt)",
      "if [[ `cat test-results/tests_exit_code.txt` -ne 0 ]]; then exit 1; fi",
  )
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  test_name = f"bite_axlearn_unit_test_tpu_{jax_version.replace('.','-') if jax_version else 'main'}"

  tpu_unittests_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=is_tpu_reserved,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=unittest_setupcmds,
      run_model_cmds=unittest_runcmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=task_owner,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/jax",
  )
  return task.run_queued_resource_test(
      task_test_config=tpu_unittests_test_config,
      task_gcp_config=job_gcp_config,
  )

def get_bite_cpu_unittests_config(
    time_out_in_min: int,
    docker_image: str,
    task_owner: str,
    jax_version: str,
    cluster: XpkClusterConfig = XpkClusters.CPU_N2_STANDARD_64_CLUSTER,
    machine_count: int = 1,
    num_slices: int = 1
    ) -> task.XpkTask:

  job_gcp_config = gcp_config.GCPConfig(
      project_name=cluster.project,
      zone=cluster.zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  unittest_setupcmds = (
    'export JAX_VERSION="0.5.3"',
  )

  # Run the unittest as non-root user, ulimit param req to mmap TPUs inside docker (default limit is 8192)
  unittest_runcmds = (
      'echo "#### Start docker image - cpu_unittests"',
      "mkdir -p test-results",
      "export JAX_ENABLE_X64=True",
      "echo gs://ml-auto-solutions/output/sparsity_diffusion_devx/axlearn-unit-tests",
      'pytest --no-header -v -m "high_cpu" --dist worksteal '
      "--csv=test-results/bite_axlearn_unit_test_cpu_high_cpu_jax_0_5_3_results.csv "
      "--csv-columns id,module,name,file,doc,markers,status,message,duration,platform,accelerator_type,datetime,test_name,jax_version "
      "--ignore axlearn/common/inference_test.py "
      "--ignore axlearn/common/flash_attention/utils_test.pym "
      "--ignore axlearn/common/flash_attention/neuron_attention_test.py || true && "
      "TESTS_EXIT_CODE=\\$? && "
      'echo "#### TPU JAX Tests finished." && '
      'echo "Test exit code is \\${TESTS_EXIT_CODE}" && '
      'echo "\\${TESTS_EXIT_CODE}" > /workspace/axlearn/test-results/tests_exit_code.txt && '
      "cp -av /workspace/axlearn/test-results /tmp_docker/ && "
      "gcloud storage cp -R test-results gs://ml-auto-solutions/output/sparsity_diffusion_devx/axlearn-unit-tests && "
      'echo "Tests exit code: \\$(cat test-results/tests_exit_code.txt)" && '
      "if [[ `cat test-results/tests_exit_code.txt` -ne 0 ]]; then exit 1; fi"
  )

  test_name = f"bite_cpu_unit_test_{jax_version.replace('.','-') if jax_version else 'main'}"

  job_test_config = test_config.CpuGkeTest(
      test_config.Cpu(
          device_type=cluster.device_version,
          machine_count=machine_count,
      ),
      test_name=test_name,
      set_up_cmds=unittest_setupcmds,
      run_model_cmds=unittest_runcmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=task_owner,
      num_slices=num_slices,
      cluster_name=cluster.name,
      docker_image=docker_image,
  )

  return task.XpkTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )

##could not get it working with existing GPU clusters
# def get_bite_gpu_unit_test_config(
#     test_name: str,
#     cluster: XpkClusterConfig,
#     docker_image: str,
#     time_out_in_min: int,
#     num_slices: int=1,
# ) -> task.XpkTask:

#   # Run the unittest as non-root user, ulimit param req to mmap TPUs inside docker (default limit is 8192)
#   unittest_runcmds = (
#       'echo "#### Start docker image - cpu_unittests"',
#       "mkdir -p test-results",
#       "export JAX_ENABLE_X64=True",
#       "echo gs://ml-auto-solutions/output/sparsity_diffusion_devx/axlearn-unit-tests",
#       'pytest --no-header -v -m "not (high_cpu or fp64 or tpu or for_8_devices or gs_login)" --test-group-count 50 --test-group 1 --dist worksteal '
#       "--csv=test-results/bite_axlearn_unit_test_cpu_high_cpu_jax_0_5_3_results.csv "
#       "--csv-columns id,module,name,file,doc,markers,status,message,duration,platform,accelerator_type,datetime,test_name,jax_version "
#       "--ignore axlearn/common/inference_test.py "
#       "--ignore axlearn/common/flash_attention/utils_test.pym "
#       "--ignore axlearn/common/flash_attention/neuron_attention_test.py || true && "
#       "TESTS_EXIT_CODE=\\$? && "
#       'echo "#### TPU JAX Tests finished." && '
#       'echo "Test exit code is \\${TESTS_EXIT_CODE}" && '
#       'echo "\\${TESTS_EXIT_CODE}" > /workspace/axlearn/test-results/tests_exit_code.txt && '
#       "cp -av /workspace/axlearn/test-results /tmp_docker/ && "
#       "gcloud storage cp -R test-results gs://ml-auto-solutions/output/sparsity_diffusion_devx/axlearn-unit-tests && "
#       'echo "Tests exit code: \\$(cat test-results/tests_exit_code.txt)" && '
#       "if [[ `cat test-results/tests_exit_code.txt` -ne 0 ]]; then exit 1; fi"
#   )

#   job_gcp_config = gcp_config.GCPConfig(
#       project_name=cluster.project,
#       zone=cluster.zone,
#       dataset_name=metric_config.DatasetOption.XLML_DATASET,
#   )

#   # test_name = f"bite_gpu_unittest_{jax_version.replace('.','-') if jax_version else 'main'}"

#   gpu_unittests_test_config = test_config.GpuXpkTest(
#       test_config.Gpu(
#           machine_type=None,
#           image_family=None,
#           count=None,
#           accelerator_type=cluster.device_version.value,
#           runtime_version=None,
#       ),
#       test_name=test_name,
#       set_up_cmds=None,
#       run_model_cmds=unittest_runcmds,
#       timeout=datetime.timedelta(minutes=time_out_in_min),
#       task_owner=test_owner.Judy_W,
#       cluster_name=cluster.name,
#       docker_image=docker_image,
#       num_slices=num_slices,
#   )

#   return task.XpkTask(
#       task_test_config=gpu_unittests_test_config,
#       task_gcp_config=job_gcp_config,
#   )



def get_bite_gpu_unittests_config(
    machine_type: str,
    image_family: str,
    count: int,
    gpu_zone: str,
    accelerator_type: str,
    runtime_version: str,
    network: str = "default",
    subnetwork: str = "default",
    # JAX version defaults to main if not specified
    jax_version: Optional[str] = None,
    project_name: Optional[Project] = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
) -> task.GpuCreateResourceTask:

  pytest_cmds=(
    """cd axlearn
export JAX_ENABLE_X64=True
pytest --no-header -v -m "not (high_cpu or fp64 or tpu or for_8_devices or gs_login)" \
--test-group-count 50 --test-group 1 --dist worksteal \
--csv=test-results/bite_axlearn_unit_test_cpu_high_cpu_jax_0_5_3_results.csv \
--csv-columns id,module,name,file,doc,markers,status,message,duration,platform,accelerator_type,datetime,test_name,jax_version \
--ignore axlearn/common/inference_test.py \
--ignore axlearn/common/flash_attention/utils_test.py \
--ignore axlearn/common/flash_attention/neuron_attention_test.py || true && \
TESTS_EXIT_CODE=\\$?
""")

# unittest_runcmds = (
#       'echo "#### Start docker image - cpu_unittests"',
#       "mkdir -p test-results",
#       "export JAX_ENABLE_X64=True",
#       "echo gs://ml-auto-solutions/output/sparsity_diffusion_devx/axlearn-unit-tests",
#       'pytest --no-header -v -m "not (high_cpu or fp64 or tpu or for_8_devices or gs_login)" --test-group-count 50 --test-group 1 --dist worksteal '
#       "--csv=test-results/bite_axlearn_unit_test_cpu_high_cpu_jax_0_5_3_results.csv "
#       "--csv-columns id,module,name,file,doc,markers,status,message,duration,platform,accelerator_type,datetime,test_name,jax_version "
#       "--ignore axlearn/common/inference_test.py "
#       "--ignore axlearn/common/flash_attention/utils_test.pym "
#       "--ignore axlearn/common/flash_attention/neuron_attention_test.py || true && "
#       "TESTS_EXIT_CODE=\\$? && "
#       'echo "#### TPU JAX Tests finished." && '
#       'echo "Test exit code is \\${TESTS_EXIT_CODE}" && '
#       'echo "\\${TESTS_EXIT_CODE}" > /workspace/axlearn/test-results/tests_exit_code.txt && '
#       "cp -av /workspace/axlearn/test-results /tmp_docker/ && "
#       "gcloud storage cp -R test-results gs://ml-auto-solutions/output/sparsity_diffusion_devx/axlearn-unit-tests && "
#       'echo "Tests exit code: \\$(cat test-results/tests_exit_code.txt)" && '
#       "if [[ `cat test-results/tests_exit_code.txt` -ne 0 ]]; then exit 1; fi"
#   )


  unittest_setupcmds = (
      # create configuration files needed
      cmd_config.dockerfile_build_gpu_cmd(),
      "nvidia-smi",
      cmd_config.pytest_env_setup_cmd(platform="gpu", jax_version=jax_version, accelerator_type=accelerator_type, pytest_cmds=pytest_cmds),
      'echo "Test exit code is \\${TESTS_EXIT_CODE}"',
      'echo "\\${TESTS_EXIT_CODE}" > /workspace/axlearn/test-results/tests_exit_code.txt',
      "cp -av /workspace/axlearn/test-results /tmp_docker/",
      "gcloud storage cp -R test-results gs://ml-auto-solutions/output/sparsity_diffusion_devx/axlearn-unit-tests",
      'echo "Tests exit code: \\$(cat test-results/tests_exit_code.txt)"',
      "if [[ `cat test-results/tests_exit_code.txt` -ne 0 ]]; then exit 1; fi",
      "chmod +x run_gpu_tests.sh",
      "cat Dockerfile_CI",
      "cat run_gpu_tests.sh",
      "sudo docker build -f Dockerfile_CI -t ml-auto-solutions/gpu_unittests .",
  )

  # Run the unittest as non-root user, ulimit param req to mmap TPUs inside docker (default limit is 8192)
  unittest_runcmds = (
    'echo "#### Start docker image - gpu_unittests"',
    "mkdir -p test-results",
    "sudo chown -R $(whoami):$(whoami) test-results",
    'sudo docker run --gpus all --shm-size="8g" --network=host --privileged --ulimit memlock=-1:-1 -v ${PWD}:/tmp_docker ml-auto-solutions/gpu_unittests  /bin/bash -c "/workspace/run_gpu_tests.sh" 2>&1 | tee test-results/tests_std_out_err.log',
    "sudo docker logs $( sudo docker ps --latest --quiet ) > test-results/docker_log.log",
    # "gcloud storage cp -R test-results/*.csv {metric_config.SshEnvVars.GCS_OUTPUT.value}axlearn-test-results/test-results",
    # "if [[ `cat test-results/tests_exit_code.txt` -ne 0 ]]; then exit 1; fi"
    )

  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=gpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  test_name = f"bite_gpu_unittest_{jax_version.replace('.','-') if jax_version else 'main'}"

  gpu_unittests_test_config = test_config.GpuVmTest(
      test_config.Gpu(
          machine_type=machine_type,
          image_family=image_family,
          count=count,
          accelerator_type=accelerator_type,
          runtime_version=runtime_version,
          network=network,
          subnetwork=subnetwork,
          attach_local_ssd=True,
          disk_size_gb=100,
      ),
      test_name=test_name,
      set_up_cmds=unittest_setupcmds,
      run_model_cmds=unittest_runcmds,
      use_existing_instance=True,
  )
  return task.GpuCreateResourceTask(
      image_family=image_family,
      image_project=ImageProject.DEEP_LEARNING_PLATFORM_RELEASE.value,
      task_test_config=gpu_unittests_test_config,
      task_gcp_config=job_gcp_config,
      install_nvidia_drivers=True,
      existing_instance_name="judyzwu-gpu-test",
      reservation=True
  )


# def get_bite_gpu_unittests_config(
#     time_out_in_min: int,
#     test_name: str,
#     cluster: XpkClusterConfig,
#     task_owner: str,
#     docker_image: str,
#     num_slices: int = 1,
# ) -> task.XpkTask:
#   pytest_cmds=("""pytest --no-header -v -m "not (high_cpu or fp64 or tpu or for_8_devices or gs_login)" \
#     --test-group-count 300 --test-group 1 --dist worksteal \
#     --ignore axlearn/common/inference_test.py \
#     --ignore axlearn/common/flash_attention/utils_test.py \
#     --ignore axlearn/common/flash_attention/neuron_attention_test.py""")

#   run_model_cmds=(
#       "ls",
#       "cd axlearn",
#       "mkdir -p test-results",
#       "pip install --upgrade pip",
#       'pip install -e ".[core,dev,gcp]"',
#       "pip install grain",
#       "pip install google-cloud-aiplatform",
#       'pytest --no-header -v -m "not (high_cpu or fp64 or tpu or for_8_devices or gs_login)" --test-group-count 300 --test-group 1 --dist worksteal --ignore axlearn/common/inference_test.py --ignore axlearn/common/flash_attention/utils_test.py',
#       "TESTS_EXIT_CODE=\$?",
#       "if [[ `cat test-results/tests_exit_code.txt` -ne 0 ]]; then exit 1; fi",
#     )

#   job_gcp_config = gcp_config.GCPConfig(
#       project_name=cluster.project,
#       zone=cluster.zone,
#       dataset_name=metric_config.DatasetOption.XLML_DATASET,
#   )

#   gpu_unittests_test_config = test_config.GpuXpkTest(
#       test_config.Gpu(
#           machine_type=None,
#           image_family=None,
#           count=None,
#           accelerator_type=cluster.device_version.value,
#           runtime_version=None,
#       ),
#       test_name=test_name,
#       set_up_cmds=None,
#       run_model_cmds=run_model_cmds,
#       timeout=datetime.timedelta(minutes=time_out_in_min),
#       task_owner=task_owner,
#       cluster_name=cluster.name,
#       docker_image=docker_image,
#       num_slices=num_slices,
#   )
#   return task.XpkTask(
#       task_test_config=gpu_unittests_test_config,
#       task_gcp_config=job_gcp_config,
#   )

