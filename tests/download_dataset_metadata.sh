#!/bin/bash

set -e

# Download TFDS metadata to flax/.tdfs/metadata directory. 
# This allows the tests to specify the `data_dir` when using tfds.testing.mock_data().
cd "$( dirname "$0" )"

if [ -d "../.tfds/metadata" ]; then 
  echo 'TFDS metadata already exists.'; 
else 
  echo 'TFDS metadata does not exist. Downloading...';
  # Note: We use subversions for faster downloads of only the metadata subdirectory
  # with no history. If this script fails for you because you don't have subversion 
  # installed, please let us know by opening a GitHub issue.
  # 
  # subversion checkout to the `trunk` branch which corresponds to `tree/master`.
  # To download from branch `foo`, replace `trunk` with `branches/foo`.
  svn checkout \
    https://github.com/tensorflow/datasets/trunk/tensorflow_datasets/testing/metadata \
    "../.tfds/metadata" -q; 
fi