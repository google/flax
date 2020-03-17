#!/bin/bash
#
# Creates a new howto diff from all files that are currently edited and not
# committed (untracked, tracked, staged). Also reverts all changes. Note the
# changes can be recovered by simply applying the diff with 
# `git apply <howto_diff>`.

# https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
# We avoid -x since it is not very useful to be verbose in this case.
set -euo

old_pwd=$(pwd)
top_dir=$(git rev-parse --show-toplevel)
howto_diff_path="${top_dir}/howtos/diffs"

cat << EOF
Awesome, you are going to create a new FLAX HOWTO!

More details about HOWTOs can be found here:
https://github.com/marcvanzee/flax/blob/prerelease/howtos/README.md

The following files are edited/added:

EOF

# Get respectively all untracked, unstaged, and stages files.
# The awk command prepends all files with "-".
(git ls-files --others --exclude-standard && \
  git diff --name-only && \
  git diff --staged --name-only) | awk '{print "- " $0}'

cat << EOF

WARNING: This operation will pack all changes in the files above into a diff 
file, and undo the changes in those files.
EOF
read -p "Would you like to continue [Y/n]? " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  printf "Bailing out."
  exit 1
fi

printf "\nPlease provide a name for the howto:\n"

read -p "howto-" howto_name
howto_name="howto-${howto_name}"
howto_path="${howto_diff_path}/${howto_name}.diff"

# Overwrite the existing diff 
if test -f "${howto_path}";  then
  printf "Diff ${howto_path} exists already.\n"
  read -p "Overwrite [Y/n]? " -r
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    printf "Bailing out."
    exit 1
  fi
fi

# Add all untracked files. After this there are no untracked changes anymore.
git add *
# Create diff for both unstaged and staged changes. Add the diff to a temporal
# location to ensure it won't be remove when we clean the changes.
tmp_path=$(mktemp)
(git diff && git diff --staged) > $tmp_path

# Revert all tracked changes (which are all edited files).
git reset --hard > /dev/null

# Add the diff file to the correct location and track it.
mv $tmp_path $howto_path && git add $howto_path

cat << EOF

Done! Diff created in ${howto_path}.
The file has been staged, you should commit and push it yourself.

NOTE: If you want to restore you changes, simply run:

$ git apply ${howto_path}
EOF

cd $old_pwd
