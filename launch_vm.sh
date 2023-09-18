#!/bin/bash

PROJECT=gcs-fuse-test
GCS_BUCKET=gcsfuse-ml-data
GCS_TFDS_BUCKET=gcsfuse-ml-data
GCS_BUCKET=princer-working-dirs
REPO=https://github.com/raj-prince/flax.git
BRANCH=princer_jax

python3 examples/cloud/launch_gce.py \
  --project=$PROJECT \
  --zone=asia-northeast1-c \
  --machine_type=a2-highgpu-2g \
  --accelerator_type=nvidia-tesla-a100 --accelerator_count=2 \
  --gcs_workdir_base=gs://$GCS_BUCKET/workdir_base \
  --tfds_data_dir=gs://$GCS_TFDS_BUCKET/datasets \
  --repo=${REPO:-https://github.com/google/flax} \
  --branch=${BRANCH:-main} \
  --example=imagenet \
  --args='--config=configs/v100_x8.py' \
  --name=princer_jax
