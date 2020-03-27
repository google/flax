#!/bin/bash
#
# Creates a HOWTO branch for each HOWTO diff file by apply the changes to the
# master branch. Removes all howto branches for which no diff file exists.

# https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo

old_pwd=$(pwd)
top_dir=$(git rev-parse --show-toplevel)
howto_diff_path="${top_dir}/howtos/diffs"
master_branch="master"

# Initialize git.
remote_repo="https://${GITHUB_ACTOR}:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git config http.sslVerify false
git config user.name "Automated Publisher"
git config user.email "actions@users.noreply.github.com"

# Fetch all branches.
git fetch --all

# Delete all remote branches starting with "howto/". This ensures we clean up
# HOWTO branches for which the diff files have been deleted.
# The sed command strips the 'origin/' prefix.
for b in $(git branch -r | grep origin/howto/ | sed 's/origin\///'); do
  git push origin --delete $branch
done

# Get names of howtos from diff files.
cd $howto_diff_path
howtos=$(ls *.diff | sed -e 's/.diff//')
cd $top_dir

printf "Applying HOWTO diffs to branches..\n"

for howto in $howtos; do
  howto_branch="howto/${howto}"
  git checkout -b $howto_branch
  diff_file="${howto_diff_path}/${howto}.diff"
  if [[ -n $(git apply --check "${diff_file}") ]]; then
    printf "\nERROR: Cannot apply howto ${howto}! ==> PLEASE FIX HOWTO\n"
    exit 1
  fi
  git apply $diff_file
  git commit -am "Added howto branch ${howto_branch}"
  git push -u origin $howto_branch
  # Make sure to checkout the master branch, otherwise the next diff branch
  # will be branched off of the current diff branch.
  git checkout $master_branch
done

cd $old_pwd
