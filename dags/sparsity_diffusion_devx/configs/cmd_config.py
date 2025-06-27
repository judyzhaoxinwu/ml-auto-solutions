# Copyright 2025 Google LLC
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

"""Utilities to construct configs for sparsify_diffusion_devx_jax_bite DAG."""

import datetime
from typing import Tuple
from dags.sparsity_diffusion_devx.configs import common

def set_up_axlearn(pinned_version, jax_version) -> Tuple[str]:
  reset_version = ""
  if pinned_version:
    reset_version = f"cd axlearn && git reset --hard {pinned_version} && cd .."

  setup_jax = None
  if jax_version:
    setup_jax = common.set_up_jax_version(jax_version)
  else:
    setup_jax = common.set_up_nightly_jax()

  return (
      common.UPGRADE_PIP,
      common.UPGRADE_SETUPTOOLS,
      common.UPGRADE_PACKAGING,
      "git clone https://github.com/apple/axlearn.git",
      reset_version,
      "python -m pip install ./axlearn[core]",
      *setup_jax,
  )


def dockerfile_build_cmd(jax_version):
  # Generate pip commands to install certain version of JAX/libTPU e.g.
  # pip install --pre jaxlib==0.5.1  -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
  # pip install jax[tpu]==0.5.1  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  # pip install jax==0.5.1
  if jax_version:
    pip_tpu_jax_install = "\n".join(
        ["RUN " + x for x in common.set_up_jax_version(jax_version)]
    )
  else:
    pip_tpu_jax_install = "\n".join(
        ["RUN " + x for x in common.set_up_nightly_jax()]
    )

  return (
      """cat > Dockerfile_CI <<EOF
FROM python:3.10-slim
WORKDIR /workspace
COPY run_tpu_tests.sh /workspace/
RUN apt update -y
RUN apt install -y git
RUN git clone https://github.com/apple/axlearn.git
WORKDIR /workspace/axlearn
RUN pip install --upgrade pip
RUN pip install -e '.[core,dev,gcp]'
RUN pip install grain
RUN pip install google-cloud-aiplatform
"""
      + pip_tpu_jax_install
      + """
RUN pip install pytest-csv
RUN pip freeze
EOF
"""
  )


def pytest_env_setup_cmd(platform,
                         jax_version,
                         accelerator_type,
                         pytest_cmds):
  return(f"""cat > run_{platform}_tests.sh <<"END_RUN_{platform.upper()}_TESTS"
  #!/bin/bash
  set -x
  echo "#### Starting {platform.upper()} JAX Tests"
  pip freeze
  cd /workspace/axlearn
  cat > conftest.py <<"END_CONFTEST_PY"
  import jax
  import os
  import pytest
  from axlearn.common.utils_spmd import setup as setup_spmd
  def pytest_sessionstart(session):
    print("Pytest session starting: Initializing JAX distributed...")
    try:
      jax.distributed.initialize()
      print("JAX distributed backend initialized successfully.")
      jax.print_environment_info()
      print("Global device count: %s", jax.device_count())
    except RuntimeError as e:
      if "backends_are_initialized" in str(e):
        print("JAX distributed backend already initialized (OK for pytest session).")
      else:
        pytest.fail("Failed to initialize JAX distributed backend: %s", e)
    except Exception as e:
      pytest.fail("Unexpected error during JAX distributed initialization: %s", e)

  def pytest_csv_register_columns(columns):
      columns["platform"] =  {platform} ### CPU, GPU, TPU
      columns["accelerator_type"] = {accelerator_type} ###
      columns["datetime"] = {datetime.datetime.now()}
      columns["test_name"] = "bite-axlearn-unit-test-{platform}-{jax_version}"
      columns["jax_version"] = {jax_version}

  END_CONFTEST_PY
  mkdir test-results
  python -c "import jax; jax.print_environment_info() ; print("Global device count: %s", jax.device_count())"
  """
  + pytest_cmds +
  f"""
  TESTS_EXIT_CODE=\\$?;echo "#### {platform.upper()} JAX Tests finished."
  """ +
  """
  echo "Test exit code is \\${TESTS_EXIT_CODE}";echo "\\${TESTS_EXIT_CODE}" > /workspace/axlearn/test-results/tests_exit_code.txt; cp -Rv /workspace/axlearn/test-results /tmp_docker/
  """ +
  f"END_RUN_{platform.upper()}_TESTS"
  )



def dockerfile_build_gpu_cmd():
  return (
      """cat > Dockerfile_CI <<EOF
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set the working directory inside the container.
WORKDIR /workspace

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        curl \
        git \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3.10-venv \
        gnupg \
        apt-transport-https \
        ca-certificates \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default `python3` command (optional, but good practice).
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set up Google Cloud SDK repository and install the SDK
RUN curl -fsSL "https://packages.cloud.google.com/apt/doc/apt-key.gpg" | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

RUN apt-get update && apt-get install -y \
    google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Clone the axlearn repository.
RUN git clone https://github.com/apple/axlearn.git

# Change the working directory to the cloned axlearn repository.
WORKDIR /workspace/axlearn

COPY conftest.py .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e '.[core,dev,gcp]' && \
    pip install --no-cache-dir grain google-cloud-aiplatform && \
    pip install --no-cache-dir \
        jax==0.5.3 \
        jaxlib==0.5.3 \
        jax[cuda12]==0.5.3 \
        triton==2.1.0 \
        nvidia-ml-py==12.560.30 \
        scipy==1.12.0 \
        numpy==1.26.4 \
        optax==0.1.7

RUN pip install --upgrade timm transformers
RUN pip install --upgrade torch torchvision torchaudio
RUN pip install pytest-test-groups
RUN pip install pytest-csv
RUN pip freeze


EOF
"""
  )
