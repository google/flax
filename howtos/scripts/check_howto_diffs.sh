#!/bin/bash
#
# Checks the following for each HOWTO diff in howtos/diffs
# 1. Check if diffs are stale and can no longer be applied to master
# 2. Apply each diff in turn and run appropriate unit tests

# https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo

old_pwd=$(pwd)
top_dir=$(git rev-parse --show-toplevel)
howto_diff_path="${top_dir}/howtos/diffs"
master_branch="origin/master"

# Get names of howtos from diff files.
cd $howto_diff_path
howtos=$(ls *.diff | sed -e 's/.diff//')
cd $top_dir

# Check for stale diffs first before running any tests. This duplicates some
# logic, but should speed up testing.
printf "Verify all HOWTO diffs can be applied...\n"

for howto in $howtos; do
  diff_file="${howto_diff_path}/${howto}.diff"

  # Check if command fails
  # See: https://stackoverflow.com/q/26675681/
  if ! git apply --check "${diff_file}"; then
    printf "\nERROR: Cannot apply howto ${howto}! ==> PLEASE FIX HOWTO\n"
    exit 1
  fi
done

printf "Run unit tests for each diff...\n"

# Fetch all branches
git fetch --all

for howto in $howtos; do
  diff_file="${howto_diff_path}/${howto}.diff"
  git apply $diff_file

  # Run unit test on affected examples only
  if ! git diff --name-only $master_branch | xargs dirname | xargs pytest; then
    printf "\nERROR: Tests failed for howto ${howto}! ==> PLEASE FIX HOWTO\n"

    # Undo patch in case we're running locally
    git apply -R $diff_file
    exit 1
  fi

  # Undo patch so we can run the next test
  git apply -R $diff_file
done

cd $old_pwd
