#!/bin/bash

PYTEST_OPTS=
RUN_DOCTEST=true
RUN_MYPY=true
RUN_PYTEST=true
RUN_PYTYPE=true
GH_VENV=false

for flag in "$@"; do
case $flag in
  --with-cov)
  PYTEST_OPTS+="--cov=flax --cov-report=xml --cov-report=term --cov-config=setup.cfg"
  ;;
  --help)
  echo "Usage:"
  echo "  --with-cov: Also generate pytest coverage."
  exit
  ;;
  --no-doctest)
  RUN_DOCTEST=false
  ;;
  --no-pytest)
  RUN_PYTEST=false
  ;;
  --no-pytype)
  RUN_PYTYPE=false
  ;;
  --no-mypy)
  RUN_MYPY=false
  ;;
  --use-venv)
  GH_VENV=true
  ;;
  *)
  echo "Unknown flag: $flag"
  exit 1
  ;;
esac
done

# Activate cached virtual env for github CI
if $GH_VENV; then
  source $(dirname "$0")/../venv/bin/activate
fi

echo "====== test config ======="
echo "PYTEST_OPTS: $PYTEST_OPTS"
echo "RUN_DOCTEST: $RUN_DOCTEST"
echo "RUN_PYTEST: $RUN_PYTEST"
echo "RUN_MYPY: $RUN_MYPY"
echo "RUN_PYTYPE: $RUN_PYTYPE"
echo "GH_VENV: $GH_VENV"
echo "WHICH PYTHON: $(which python)"
echo "jax: $(python -c 'import jax; print(jax.__version__)')"
echo "flax: $(python -c 'import flax; print(flax.__version__)')"
echo "=========================="
echo ""

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

# Run embedded tests inside docs
if $RUN_DOCTEST; then
  echo "=== RUNNING DOCTESTS ==="
  # test doctest
  sphinx-build -M doctest docs docs/_build -T
  # test build html
  sphinx-build -M html docs docs/_build -T
  # test docstrings
  pytest -n auto flax --doctest-modules --suppress-no-test-exit-code
fi

# check that flax is running on editable mode
# (i.e. no notebook installed flax from pypi)
echo "=== CHECKING FLAX IS EDITABLE ==="
assert_error="flax is not running on editable mode."
(cd docs; python -c "import flax; assert 'site-packages' not in flax.__file__, \"$assert_error\"")

# env vars must be set after doctest
export JAX_NUMPY_RANK_PROMOTION=raise
export FLAX_PROFILE=1
export FLAX_LAZY_RNG=1

if $RUN_PYTEST; then
  echo "=== RUNNING PYTESTS ==="
  # Run some test on separate process, avoiding device configs poluting each other
  PYTEST_IGNORE=
  for file in "tests/jax_utils_test.py"; do
      echo "pytest -n auto $file $PYTEST_OPTS"
      pytest -n auto $file $PYTEST_OPTS
      PYTEST_IGNORE+=" --ignore=$file"
  done
  # Run battery of core FLAX API tests.
  echo "pytest -n auto tests $PYTEST_OPTS $PYTEST_IGNORE"
  pytest -n auto tests $PYTEST_OPTS $PYTEST_IGNORE

  # Per-example tests.
  #
  # we apply pytest within each example to avoid pytest's annoying test-filename collision.
  # In pytest foo/bar/baz_test.py and baz/bleep/baz_test.py will collide and error out when
  # /foo/bar and /baz/bleep aren't set up as packages.
  for egd in $(find examples -maxdepth 1 -mindepth 1 -type d); do
      pytest $egd
  done
fi

if $RUN_PYTYPE; then
  echo "=== RUNNING PYTYPE ==="
  # Validate types in library code.
  pytype --jobs auto --config pytype.cfg flax/

  # Validate types in examples.
  for egd in $(find examples -maxdepth 1 -mindepth 1 -type d); do
      # use cd to make sure pytpe cache lives in example dir and doesn't name clash
      # use *.py to avoid importing configs as a top-level import which leads to import errors
      # because config files use relative imports (e.g. from config import ...).
      (cd $egd ; pytype --jobs auto --config ../../pytype.cfg "*.py")
  done
fi

if $RUN_MYPY; then
  echo "=== RUNNING MYPY ==="
  # Validate types in library code.
  mypy --config mypy.ini flax/ --show-error-codes
fi

# Return error code 0 if no real failures happened.
echo "finished all tests."
