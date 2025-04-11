
Flax
====
.. div:: sd-text-left sd-font-italic

   **N**\ eural **N**\ etworks for JA\ **X**


----

Flax provides a **flexible end-to-end user experience for researchers and developers who use JAX for neural networks**. Flax enables you to use the full power of `JAX <https://jax.readthedocs.io>`__.

At the core of Flax is **NNX - a simplified API that makes it easier to create, inspect,
debug, and analyze neural networks in JAX.** Flax NNX has first class support
for Python reference semantics, enabling users to express their models using regular
Python objects. Flax NNX is an evolution of the previous `Flax Linen <https://flax-linen.readthedocs.io/>`__
API, and it took years of experience to bring a simpler and more user-friendly API.

.. note::
   Flax Linen API is not going to be deprecated in the near future as most of Flax users still rely on this API. However, new users are encouraged to use Flax NNX. Check out `Why Flax NNX <why.html>`_ for a comparison between Flax NNX and Linen, and our reasoning to make the new API.

   To move your Flax Linen codebase to Flax NNX, get familiarized with the API in `NNX Basics <https://flax.readthedocs.io/en/latest/nnx_basics.html>`_ and then start your move following the `evolution guide <guides/linen_to_nnx.html>`_.

Features
^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Pythonic
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Flax NNX supports the use of regular Python objects, providing an intuitive
            and predictable development experience.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Simple
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Flax NNX relies on Python's object model, which results in simplicity for
            the user and increases development speed.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Expressive
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Flax NNX allows fine-grained control of the model's state via
            its `Filter <https://flax.readthedocs.io/en/latest/guides/filters_guide.html>`__
            system.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Familiar
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Flax NNX makes it very easy to integrate objects with regular JAX code
            via the `Functional API <nnx_basics.html#the-flax-functional-api>`__.

Basic usage
^^^^^^^^^^^^

.. testsetup::

   import jax
   import jax.numpy as jnp

.. testcode::

   from flax import nnx
   import optax


   class Model(nnx.Module):
     def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
       self.linear = nnx.Linear(din, dmid, rngs=rngs)
       self.bn = nnx.BatchNorm(dmid, rngs=rngs)
       self.dropout = nnx.Dropout(0.2, rngs=rngs)
       self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

     def __call__(self, x):
       x = nnx.relu(self.dropout(self.bn(self.linear(x))))
       return self.linear_out(x)

   model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # eager initialization
   optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # reference sharing

   @nnx.jit  # automatic state management for JAX transforms
   def train_step(model, optimizer, x, y):
     def loss_fn(model):
       y_pred = model(x)  # call methods directly
       return ((y_pred - y) ** 2).mean()

     loss, grads = nnx.value_and_grad(loss_fn)(model)
     optimizer.update(grads)  # in-place updates

     return loss


Installation
^^^^^^^^^^^^

Install via pip:

.. code-block:: bash

   pip install flax

Or install the latest version from the repository:

.. code-block:: bash

   pip install git+https://github.com/google/flax.git


----

Learn more
^^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` Flax NNX Basics
         :class-card: sd-text-black sd-bg-light
         :link: nnx_basics.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` MNIST Tutorial
         :class-card: sd-text-black sd-bg-light
         :link: mnist_tutorial.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` Guides
         :class-card: sd-text-black sd-bg-light
         :link: guides/index.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`transform;2em` Flax Linen to Flax NNX
         :class-card: sd-text-black sd-bg-light
         :link: guides/linen_to_nnx.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`menu_book;2em` API reference
         :class-card: sd-text-black sd-bg-light
         :link: api_reference/index.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`import_contacts;2em` Glossary
         :class-card: sd-text-black sd-bg-light
         :link: nnx_glossary.html


----

.. toctree::
   :hidden:
   :maxdepth: 2

   nnx_basics
   mnist_tutorial
   why
   guides/index
   examples/index
   nnx_glossary
   The Flax philosophy <https://flax.readthedocs.io/en/latest/philosophy.html>
   How to contribute <https://flax.readthedocs.io/en/latest/contributing.html>
   api_reference/index
