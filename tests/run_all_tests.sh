#!/bin/bash

# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
PYTEST_OPTS=
RUN_DOCTEST=false
RUN_MYPY=false
RUN_PYTEST=false
RUN_PYTYPE=false
GH_VENV=false

for flag in "$@"; do
case $flag in
  --with-cov)
  PYTEST_OPTS+="--cov=flax --cov-report=xml --cov-report=term --cov-config=pyproject.toml"
  ;;
  --help)
  echo "Usage:"
  echo "  --with-cov: Also generate pytest coverage."
  exit
  ;;
  --only-doctest)
  RUN_DOCTEST=true
  ;;
  --only-pytest)
  RUN_PYTEST=true
  ;;
  --only-pytype)
  RUN_PYTYPE=true
  ;;
  --only-mypy)
  RUN_MYPY=true
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

# if neither --only-doctest, --only-pytest, --only-pytype, --only-mypy is set, run all tests
if ! $RUN_DOCTEST && ! $RUN_PYTEST && ! $RUN_PYTYPE && ! $RUN_MYPY; then
  RUN_DOCTEST=true
  RUN_PYTEST=true
  RUN_PYTYPE=true
  RUN_MYPY=true
fi

# Activate cached virtual env for github CI
if $GH_VENV; then
  source $(dirname "$0")/../.venv/bin/activate
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
  sphinx-build -M doctest docs_nnx docs_nnx/_build -T
  # test build html
  sphinx-build -M html docs docs/_build -T
  sphinx-build -M html docs_nnx docs_nnx/_build -T
  # test docstrings
  pytest -n auto flax \
    --doctest-modules \
    --suppress-no-test-exit-code \
    --ignore=flax/nnx/examples
fi

# check that flax is running on editable mode
# (i.e. no notebook installed flax from pypi)
echo "=== CHECKING FLAX IS EDITABLE ==="
assert_error="flax is not running on editable mode."
(cd docs; python -c "import flax; assert 'site-packages' not in flax.__file__, \"$assert_error\"")

# env vars must be set after doctest
export JAX_NUMPY_RANK_PROMOTION=raise
export FLAX_PROFILE=1

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
  # Run nnx tests
  pytest -n auto flax/nnx/tests $PYTEST_OPTS $PYTEST_IGNORE
  pytest -n auto docs/_ext/codediff_test.py $PYTEST_OPTS $PYTEST_IGNORE

  # Per-example tests.
  #
  # we apply pytest within each example to avoid pytest's annoying test-filename collision.
  # In pytest foo/bar/baz_test.py and baz/bleep/baz_test.py will collide and error out when
  # /foo/bar and /baz/bleep aren't set up as packages.
  for egd in $(find examples -maxdepth 1 -mindepth 1 -type d); do
    # skip if folder starts with "_"
    if [[ $egd == *"_"* ]]; then
      continue
    fi
    pytest $egd
  done

  for egd in $(find flax/nnx/examples -maxdepth 1 -mindepth 1 -type d); do
    # skip if folder starts with "_" or is "toy_examples"
    if [[ $egd == *"_"* ]] || [[ $egd == *"toy_examples"* ]]; then
      continue
    fi
    pytest $egd
  done
fi

if $RUN_PYTYPE; then
  echo "=== RUNNING PYTYPE ==="
  # Validate types in examples.
  for egd in $(find examples -maxdepth 1 -mindepth 1 -type d); do
    # skip if folder starts with "_" or is "nnx_toy_examples"
    if [[ $egd == *"_"* ]] || [[ $egd == *"nnx_toy_examples"* ]]; then
      continue
    fi
    # use cd to make sure pytype cache lives in example dir and doesn't name clash
    # use *.py to avoid importing configs as a top-level import which leads to import errors
    # because config files use relative imports (e.g. from config import ...).
    (cd $egd ; pytype "*.py" --jobs auto --config ../../pyproject.toml)
  done
  # Validate types in library code.
  pytype --jobs auto --config pyproject.toml flax/
fi

if $RUN_MYPY; then
  echo "=== RUNNING MYPY ==="
  # Validate types in library code.
  mypy --config pyproject.toml flax/ --show-error-codes
fi

# Return error code 0 if no real failures happened.
echo "finished all tests."
