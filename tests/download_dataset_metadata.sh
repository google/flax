#!/bin/bash


set -e

# Download TFDS metadata to flax/.tdfs/metadata directory.
# This allows the tests to specify the `data_dir` when using tfds.testing.mock_data().
cd "$( dirname "$0" )"

if [ -d "../.tfds/metadata" ]; then
  echo 'TFDS metadata already exists.';
else
  echo 'TFDS metadata does not exist. Downloading...';
  git clone --depth 3  --filter=blob:none   --sparse   https://github.com/tensorflow/datasets/
  cd datasets
  git sparse-checkout set tensorflow_datasets/testing/metadata
  mkdir ../../.tfds
  mv tensorflow_datasets/testing/metadata/ ../../.tfds/metadata/
  cd ..
  rm -rf datasets
fi
