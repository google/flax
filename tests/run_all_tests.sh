#!/bin/bash

# Download TFDS metadata to flax/.tdfs/metadata directory. 
# This allows the tests to specify the `data_dir` when using tfds.testing.mock_data().

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)"
DIR="$( dirname "${DIR}" )"

if [ -d "${DIR}/.tfds/metadata" ]; then 
  echo 'Exists'; 
else 
  echo 'TFDS metadata doesnt exist. Downloading...';
  # subversion checkout to the `trunk` branch which corresponds to `tree/master`.
  # To download from branch `foo`, replace `trunk` with `branches/foo`.
  svn checkout \
    https://github.com/tensorflow/datasets/trunk/tensorflow_datasets/testing/metadata \
    "${DIR}/.tfds/metadata" -q; 
fi

# Instead of using set -e, we have a manual error trap that
# exits for any error code != 5 since pytest returns error code 5
# for no found tests. (We may force minimal test coverage in examples
# in the future!)
trap handle_errors ERR
handle_errors () {
    ret="$?"
    if [[ "$ret" == 5 ]]; then
      echo "error code $ret == no tests found in $egd"
    else
      echo "error code $ret"
      exit 1
    fi
}

# Run battery of core FLAX API tests.
PYTEST_OPTS=
if [[ $1 == "--with-cov" ]]; then
    PYTEST_OPTS+="--cov=flax --cov-report=xml --cov-report=term --cov-config=setup.cfg"
fi
pytest -n 4 tests $PYTEST_OPTS

# Per-example tests.
# we apply pytest within each example to avoid pytest's annoying test-filename collision.
# In pytest foo/bar/baz_test.py and baz/bleep/baz_test.py will collide and error out when
# /foo/bar and /baz/bleep aren't set up as packages.
for egd in $(find examples -maxdepth 1 -mindepth 1 -type d); do
    pytest $egd
done

# Return error code 0 if no real failures happened.
echo "finished all tests."
