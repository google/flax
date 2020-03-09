#!/bin/sh
#
# Creates a new howto diff from all files that are currently edited and not
# committed (untracked, tracked, staged). Also reverts all changes. Note the
# changes can be recovered by simply applying the diff with 
# `git apply <howto_diff>`.

. howtos/scripts/common.sh

cd $top_dir

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
read -p "Would you like to continue? " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  printf "Bailing out."
  exit 1
fi

printf "\nPlease provide a name for the howto:\n"

read -p "howto-" howto_name
howto_name="howto-${howto_name}"
howto_path="${howto_diff_path}/${howto_name}.diff"

if test -f "${howto_path}"; then
    printf "ERROR: howto exists already at ${howto_path}"
    printf "Please provide another howto name!"
    exit 1
fi

# Add all untracked files. After this there are no untracked changes anymore.
git add *
# Create diff for both unstaged and staged changes.
(git diff && git diff --staged) > $howto_path

# Revert all tracked changes (which are all files except the howto diff).
git reset --hard > /dev/null

# Make sure the diff is tracked.
git add $howto_path

cat << EOF

Done! Diff created in ${howto_path}.
The file has been staged, you should commit and push it yourself.

NOTE: If you want to restore you changes, simply run:

$ git apply ${howto_path}
EOF

cd $old_pwd