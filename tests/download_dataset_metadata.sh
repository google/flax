#!/bin/bash
# If you get an error like:
#   Cloning into 'datasets'...
#   fatal: cannot change to 'https://github.com/tensorflow/datasets/': No such file or directory
#   error: failed to initialize sparse-checkout
# This mean your git version is outdated. Just update it.


set -e

# Download TFDS metadata to flax/.tdfs/metadata directory.
# This allows the tests to specify the `data_dir` when using tfds.testing.mock_data().
cd "$( dirname "$0" )"

if [ -d "../.tfds/metadata" ]; then
  echo 'TFDS metadata already exists.';
else
  echo 'TFDS metadata does not exist. Downloading...';
  git clone --branch v4.8.2 --depth 3 --filter=blob:none --sparse https://github.com/tensorflow/datasets/
  cd datasets
  git sparse-checkout set tensorflow_datasets/testing/metadata
  mkdir ../../.tfds
  mv tensorflow_datasets/testing/metadata/ ../../.tfds/metadata/
  cd ..
  rm -rf datasets
fi
