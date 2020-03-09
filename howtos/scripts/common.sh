#!/bin/bash

# Fail on any error and be verbose.
set -ex

old_pwd=$(pwd)

master_branch="prerelease"

top_dir=$(git rev-parse --show-toplevel)

howto_diff_path="${top_dir}/howtos/diffs"

cd $top_dir
