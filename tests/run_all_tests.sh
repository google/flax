#!/bin/bash

export FLAX_PROFILE=1

ALL_EXAMPLES=false
PYTEST_OPTS=
for flag in "$@"; do
case $flag in
  --all)
  ALL_EXAMPLES=true
  ;;
  --with-cov)
  PYTEST_OPTS+="--cov=flax --cov-report=xml --cov-report=term --cov-config=setup.cfg"
  ;;
  --help)
  echo "Usage:"
  echo "  --all: Also run tests for deprecated examples."
  echo "  --with-cov: Also generate pytest coverage."
  exit
  ;;
  *)
  echo "Unknown flag: $flag"
  exit 1
  ;;
esac
done

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
pytest -n 4 tests $PYTEST_OPTS

# validate types
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Pytype is currently not working on MacOS, see https://github.com/google/pytype/issues/661"
else
  pytype flax/
fi

# Per-example tests.
if [[ $ALL_EXAMPLES == 'true' ]]; then
  # we apply pytest within each example to avoid pytest's annoying test-filename collision.
  # In pytest foo/bar/baz_test.py and baz/bleep/baz_test.py will collide and error out when
  # /foo/bar and /baz/bleep aren't set up as packages.
  for egd in $(find examples -maxdepth 1 -mindepth 1 -type d); do
      pytest $egd
  done
fi

# Per-example tests for Linen examples.
for egd in $(find linen_examples -maxdepth 1 -mindepth 1 -type d); do
    pytest $egd
    # use cd to make sure pytpe cache lives in example dir and doesn't name clash
    # use *.py to avoid importing configs as a top-level import which leads tot import errors
    # because config files use relative imports (e.g. from config import ...). 
    (cd $egd ; pytype "*.py")
done

# Return error code 0 if no real failures happened.
echo "finished all tests."
