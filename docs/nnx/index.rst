
NNX
========


NNX is a **N**\ eural **N**\ etwork library for JA\ **X** that focuses on providing the best
development experience, so building and experimenting with neural networks is easy and
intuitive. It achieves this by embracing Python’s object-oriented model and making it
compatible with JAX transforms, resulting in code that is easy to inspect, debug, and
analyze.

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

            NNX supports the use of regular Python objects, providing an intuitive
            and predictable development experience.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Simple
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            NNX relies on Python's object model, which results in simplicity for
            the user and increases development speed.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Streamlined
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            NNX integrates user feedback and hands-on experience with Linen
            into a new simplified API.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Compatible
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            NNX makes it very easy to integrate objects with regular JAX code
            via the `Functional API <nnx_basics.html#the-functional-api>`__.

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

   @nnx.jit # automatic state management
   def train_step(model, optimizer, x, y):
     def loss_fn(model):
       y_pred = model(x)  # call methods directly
       return ((y_pred - y) ** 2).mean()

     loss, grads = nnx.value_and_grad(loss_fn)(model)
     optimizer.update(grads)  # inplace updates

     return loss


Installation
^^^^^^^^^^^^

Install NNX via pip:

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

      .. card:: :material-regular:`rocket_launch;2em` NNX Basics
         :class-card: sd-text-black sd-bg-light
         :link: nnx_basics.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` MNIST Tutorial
         :class-card: sd-text-black sd-bg-light
         :link: mnist_tutorial.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`sync_alt;2em` NNX vs JAX Transformations
         :class-card: sd-text-black sd-bg-light
         :link: transforms.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`transform;2em` Haiku and Linen vs NNX
         :class-card: sd-text-black sd-bg-light
         :link: haiku_linen_vs_nnx.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`menu_book;2em` API reference
         :class-card: sd-text-black sd-bg-light
         :link: ../api_reference/flax.nnx/index.html


----

.. toctree::
   :hidden:
   :maxdepth: 1

   haiku_linen_vs_nnx
   nnx_basics
   mnist_tutorial
   transforms