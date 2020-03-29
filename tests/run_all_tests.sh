#!/bin/bash

pytest -n 4 tests -W ignore

# we apply pytest within each example to avoid pytest's annoying test-filename collision.
# In pytest foo/bar/baz_test.py and baz/bleep/baz_test.py will collide and error out when
# /foo/bar and /baz/bleep aren't set up as packages.
for egd in $(find examples -maxdepth 1 -mindepth 1 -type d); do
    pytest $egd -W ignore
    # Pytest returns error code 5 for no found tests, but don't forward this as an error.
    # We may remove this to force minimal test coverage in the future!
    ret=$?
    if [[ "$ret" != 0 ]] && [[ "$ret" != 5 ]]; then exit 1; fi
done
