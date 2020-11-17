# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Getting started

Currently, you need to install Python 3.6 for developing Flax, and `svn` for running the `run_all_tests.sh` script. After installing these prerequisites, you can clone the repository, set up your local environment, and run all tests with the following commands:

```
git clone https://github.com/google/flax
cd flax
python3.6 -m virtualenv env
. env/bin/activate
pip install -e . .[testing]
./tests/run_all_tests.sh
```

Alternatively, you can also develop inside a Docker container : See [`dev/README.md`](https://github.com/google/flax/blob/master/dev/README.md).

We welcome pull requests, in particular for those issues [marked as PR-ready](https://github.com/google/flax/issues?q=is%3Aopen+is%3Aissue+label%3A%22Status%3A+pull+requests+welcome%22). For other proposals, we ask that you first open an Issue to discuss your planned contribution.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).
