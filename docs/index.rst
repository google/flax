.. Flax documentation main file, created by
   sphinx-quickstart on Mon Feb 17 11:41:38 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Flax: Neural networks with JAX
====

Flax delivers an **end-to-end, flexible, user experience for researchers
who use JAX with neural networks**. Flax exposes the full power of JAX.
It is made up of loosely coupled libraries,
which are showcased with end-to-end integrated guides and examples. |emphasized hyperlink|_

.. |emphasized hyperlink| replace:: **Get started now ▶**
.. _emphasized hyperlink: getting_started.html

.. code:: python

   class MLP(nn.Module):
     @nn.compact
     def __call__(self, x):
       x = nn.Dense(16)(x)
       x = nn.relu(x)
       x = nn.Dense(16)(x)
       return x

   model = MLP()
   initialized_variables = model.init({"params": PRNGKey(42)})
   model.apply(initialized_variables, jnp.ones((4, 16)))

----

.. list-table::
   :align: center

   * - `Getting started ▶ <getting_started.html>`__

       JAX, Flax, and how to use them

     - `Guides ▶ <getting_started.html>`__

       ...

     - `End-to-end examples ▶ <getting_started.html>`__

       ...

     - `How Flax works ▶ <getting_started.html>`__

       ...

     - `API reference docs ▶ <getting_started.html>`__

       ...

----

..
  TODO: Make all links open in a new tab

Flax is used by `hundreds of projects (and growing) <https://github.com/google/flax/network/dependents?package_id=UGFja2FnZS01MjEyMjA2MA%3D%3D>`__,
both in the open source community and within Google.
Notable examples include:

- `🤗 Hugging Face <https://huggingface.co/transformers/#supported-frameworks>`__ for NLP
- `NetKet <https://github.com/netket/netket>`__ for Quantum ML
- `T5x <https://github.com/google-research/t5x>`__ for large language models
- `Brax <https://github.com/google/brax>`__ for on-device differentiable 
  robotics simulations.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

   overview
   installation
   Examples <https://github.com/google/flax/tree/main/examples>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Guided Tour

   notebooks/jax_for_the_impatient
   notebooks/flax_basics
   notebooks/annotated_mnist

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: How do I ...?
   :glob:
   :titlesonly:

   howtos/state_params
   howtos/ensembling
   howtos/lr_schedule
   howtos/extracting_intermediates
   howtos/model_surgery

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Design Notes
   :glob:
   :titlesonly:

   design_notes/*
   FLIPs <https://github.com/google/flax/tree/main/docs/flip>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Additional material

   philosophy
   contributing

.. toctree::
   :hidden:

   :maxdepth: 2
   :caption: API reference

   flax.linen
   flax.optim
   flax.serialization
   flax.core.frozen_dict
   flax.struct
   flax.jax_utils
   flax.traceback_util
   flax.traverse_util
   flax.training
   flax.config
   flax.errors

.. toctree::
   :hidden:

   :maxdepth: 1
   :caption: (deprecated)

   flax.nn (deprecated) <flax.nn>
