# Where to find the docs

The FLAX documentation can be found here:
https://flax.readthedocs.io/en/latest/

# How to build the docs

1. Install the requirements using `pip install -r docs/requirements.txt`
2. Make sure `pandoc` is installed
3. Run the make script `make html`

# How to run embedded code tests

We use `doctest` blocks for embedded code in documents, that are also
tested. Learn more at https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html

To run tests locally, run `make doctest`
