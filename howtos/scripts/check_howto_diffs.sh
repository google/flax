#!/bin/bash
#
# Checks the following for each HOWTO diff in howtos/diffs:
# 1. Check if diffs are stale and can no longer be applied to *this branch*
# 2. Apply each diff in turn and run appropriate unit tests in *this branch*

# TODO (dsuo): refactor duplicate code from check, pack, and apply_howto_diffs.

# https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo

old_pwd=$(pwd)
top_dir=$(git rev-parse --show-toplevel)
howto_diff_path="${top_dir}/howtos/diffs"

# Get names of howtos from diff files.
cd $howto_diff_path
howtos=$(ls *.diff | sed -e 's/.diff//')
cd $top_dir

# Check for stale diffs first before running any tests. This duplicates some
# logic, but should speed up testing.
printf "Verifying all HOWTO diffs can be applied...\n"

for howto in $howtos; do
  diff_file="${howto_diff_path}/${howto}.diff"

  # Check if command fails
  # See: https://stackoverflow.com/q/26675681/
  # NOTE: This check will pass if the branches can be merged with conflicts
  if ! git apply --check --3way "${diff_file}"; then
    printf "\nERROR: Cannot apply howto ${howto}! ==> PLEASE FIX HOWTO\n"
    exit 1
  fi
done

printf "Running unit tests for each diff...\n"

for howto in $howtos; do
  diff_file="${howto_diff_path}/${howto}.diff"

  # NOTE: If there is a merge conflict, we fail and leave conflicts in the
  # local copy. If the user/developer's local copy was dirty, this may cause
  # some pain. Ideally, we would want to check if applying a patch would cause
  # merge conflicts _before_ applying the patch. So far, we have found three
  # non-ideal ways to avoid applying conflicting merges:
  #
  # 1. Demand a clean local copy before running this script. This solution
  # inevitably leads to many unnecessary commits since each time we run the
  # script, we find one new error, but we must commit the fix before we can run
  # the script again.
  # 2. Create a new branch and use `git merge --no-commit`. If demanding a
  # clean local copy is unreasonable, then we need to contend with many
  # possible states a local copy may be in. Creating a new branch gets dicey
  # quickly.
  # 3. Parse output from `git apply --check --3way`. This is the least bad way
  # (so far), but it's also kludgy. `git apply --check --3way` outputs to
  # a TTY, not to stderr or stdout directly, so we have to run something likke
  # the line below to check for conflicts without applying the patch:
  #
  #  > script -c '(git apply --check --3way howtos/diffs/checkpointing.diff >
  #    $(tty))' | grep conflicts
  #
  # And so, we leave the merge conflicts for now.
  git apply --3way $diff_file

  # Run unit test on affected examples only.
  # NOTE: this will pick up dirty changes outside of the patch as well
  if ! git diff --name-only HEAD | xargs dirname | xargs pytest; then
    printf "\nERROR: Tests failed for howto ${howto}! ==> PLEASE FIX HOWTO\n"

    # NOTE: Originally, we undid changes, but now we leave local copy modified
    # on error
    exit 1
  fi

  # Undo patch so we can run the next test.
  # TODO (dsuo): reversing a three-way merge on a (potentially) dirty local
  # copy is a bit of a minefield
  git reset --hard HEAD
done

cd $old_pwd
