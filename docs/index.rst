.. Flax documentation master file, created by
   sphinx-quickstart on Mon Feb 17 11:41:38 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Flax documentation
==================

Flax is a neural network library for JAX that is designed for flexibility and
is in use by a growing community
of researchers and engineers at Google who happily use Flax for their
daily research. The new Flax `"Linen" module API <https://github.com/google/flax/tree/master/flax/linen>`_ is 
now stable and we recommend it for all new projects. The old `flax.nn` API will be deprecated. Please report
any feature requests, issues, questions or concerns in our 
`discussion forum <https://github.com/google/flax/discussions>`_ , or just let us know 
what you're working on!

Expect changes to the
API, but we'll use deprecation warnings when we can, and keep
track of them in our `Changelog <https://github.com/google/flax/CHANGELOG.md>`_.

In case you need to reach us directly, we're at flax-dev@google.com.

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   overview
   installation
   notebooks/index
   examples

.. toctree::
   :maxdepth: 1
   :caption: Flax design notes

   patterns/flax_patterns
   design_notes/design_notes

.. toctree::
   :maxdepth: 1
   :caption: Additional material

   philosophy
   contributing
   faq

.. toctree::
   :maxdepth: 2
   :caption: API reference

   flax.linen
   flax.optim
   flax.serialization
   flax.struct
   flax.jax_utils
   flax.nn (deprecated) <flax.nn>

