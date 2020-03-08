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
git remote add publisher "${remote_repo}"
git show-ref # useful for debugging
git branch --verbose

# publish any new files
# git checkout master
# git add -A
# timestamp=$(date -u)
# git commit -m "Automated publish: ${timestamp} ${GITHUB_SHA}" || exit 0
# git pull --rebase publisher master
# git push publisher master

git checkout prerelease

howto_diff_path="howtos/diffs"

top_dir=$(git rev-parse --show-toplevel)

cd "${top_dir}/${howto_diff_path}"
howtos=$(ls *.diff | sed -e 's/.diff//')
cd $top_dir

echo "Applying HOWTO diffs to branches..\n"
for howto in $howtos; do
  echo $howto
  if [[ -n $(git rev-parse --verify --quiet $howto) ]]; then
    echo "  Branch ${howto} exists already: overriding with diff"
    git branch -D $howto
  fi
  git checkout -b $howto
  git apply "howtos/${howto}.diff"
  git commit -am "Added howto branch ${howto}"
  git push
done



