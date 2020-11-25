#!/bin/bash

export FLAX_PROFILE=1

sh $(dirname "$0")/download_dataset_metadata.sh || exit

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

# validate types
pytype flax/

# Per-example tests.
# we apply pytest within each example to avoid pytest's annoying test-filename collision.
# In pytest foo/bar/baz_test.py and baz/bleep/baz_test.py will collide and error out when
# /foo/bar and /baz/bleep aren't set up as packages.
for egd in $(find examples -maxdepth 1 -mindepth 1 -type d); do
    pytest $egd
done

# Per-example tests for linen examples.
for egd in $(find linen_examples -maxdepth 1 -mindepth 1 -type d); do
    pytest $egd
done

# Return error code 0 if no real failures happened.
echo "finished all tests."
