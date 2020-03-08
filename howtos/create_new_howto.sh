#!/bin/sh
# Creates a new diff howto from the current open files.

old_pwd=$(pwd)

# Change to top of repository.
cd $(git rev-parse --show-toplevel)

howto_diff_path="$(pwd)/howtos/diffs"

cat << EOF
Awesome, you are going to create a new FLAX HOWTO!

More details about HOWTOs can be found here:
https://github.com/marcvanzee/flax/blob/prerelease/howtos/README.md

The following files are edited/added:

EOF
git diff --name-only | awk '{print "- " $0}'
git diff --staged --name-only | awk '{print "- " $0}'
git ls-files --others --exclude-standard

echo ""
echo "This operation will pack all changes in the files above into a diff "
echo "file, and undo the changes in those files."
read -p "Would you like to continue? " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "\nBailing out."
  exit 1
fi

echo "\nPlease provide a name for the howto:\n"

read -p "howto-" howto_name
howto_name="howto-${howto_name}"
howto_path="${howto_diff_path}/${howto_name}.diff"

if test -f "${howto_path}"; then
    echo "ERROR: howto exists already at ${howto_path}"
    echo "Please provide another howto name!"
    exit 1
fi

# Add all untracked files
git add *
# Store the diff in a temporary location.
tmp_path=$(mktemp)
git diff > $tmp_path
# Also add staged changes.
git diff --staged >> $tmp_path

# Now undo all changes that are made.
git restore --tracked <path>
git restore <path>
git clean -f -- howtos/howto-setupchange.diff

# Make sure the diff is tracked.
git add $howto_path
echo "\nDone! Diff created in ${howto_path}"
echo "The file has been added to your index automatically."





cd $old_pwd
