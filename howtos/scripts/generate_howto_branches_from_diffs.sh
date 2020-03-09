#!/bin/sh

set -ex

# check values
if [ -z "${GITHUB_TOKEN}" ]; then
    echo "error: GITHUB_TOKEN not found"
    exit 1
fi

# initialize git
remote_repo="https://${GITHUB_ACTOR}:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git config http.sslVerify false
git config user.name "Automated Publisher"
git config user.email "actions@users.noreply.github.com"


# Fetch all branches
git fetch --no-tags --prune --depth=1 origin +refs/heads/*:refs/remotes/origin/*

# git remote add origin "${remote_repo}"
# git show-ref # useful for debugging
# git branch --verbose
# git branch -a
# echo "Does this work?"

# publish any new files
# git checkout master
# git add -A
# timestamp=$(date -u)
# git commit -m "Automated publish: ${timestamp} ${GITHUB_SHA}" || exit 0
# git pull --rebase publisher master
# git push publisher master

git checkout prerelease

# First delete all remote branches starting with "howto-"
git branch -a
for b in $(git branch -r | grep origin/howto); do
  branch=${b##*/}
  git push origin --delete $branch
done

howto_diff_path="howtos/diffs"

top_dir=$(git rev-parse --show-toplevel)

cd "${top_dir}/${howto_diff_path}"
howtos=$(ls *.diff | sed -e 's/.diff//')
cd $top_dir

echo "Applying HOWTO diffs to branches..\n"
for howto in $howtos; do
  git checkout -b $howto
  diff_file="${howto_diff_path}/${howto}.diff"
  if [[ -n $(git apply --check "${diff_file}") ]]; then
    echo "ERROR: Cannot apply ${howto}!"
    exit 1
  fi
  git apply $diff_file
  git commit -am "Added howto branch ${howto}"
  git push -u origin $howto
done