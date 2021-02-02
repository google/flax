.. Flax documentation master file, created by
   sphinx-quickstart on Mon Feb 17 11:41:38 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Flax documentation
==================

Flax is a neural network library and ecosystem for JAX that is
designed for flexibility. Flax is in use by a growing community of
researchers and engineers at Google who happily use Flax for their
daily research.

For a quick introduction and short example snippets, see our `README
<https://github.com/google/flax/blob/master/README.md>`_.

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   overview
   installation
   examples

.. toctree::
   :maxdepth: 1
   :caption: Guided Tour

   notebooks/jax_for_the_impatient
   notebooks/flax_basics

.. toctree::
   :maxdepth: 1
   :caption: How do I ...?
   :glob:
   :titlesonly:

   howtos/*

.. toctree::
   :maxdepth: 1
   :caption: Design Notes
   :glob:
   :titlesonly:

   design_notes/*

.. toctree::
   :maxdepth: 1
   :caption: Additional material

   philosophy
   contributing

.. toctree::
   :maxdepth: 2
   :caption: API reference

   flax.linen
   flax.optim
   flax.serialization
   flax.struct
   flax.jax_utils
   flax.traverse_util
   flax.errors

.. toctree::
   :maxdepth: 1
   :caption: (deprecated)

   flax.nn (deprecated) <flax.nn>
