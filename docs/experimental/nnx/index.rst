
NNX
========


NNX is a JAX-based neural network library designed for simplicity and power. Its modular
approach follows standard Python conventions, making it both intuitive and compatible with
the broader JAX ecosystem.

.. note::
   NNX is currently in an experimental state and is subject to change. Linen is still the
   recommended option for large-scale projects. Feedback and contributions are welcome!

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

            Modules are standard Python classes, promoting ease of use and a more familiar
            development experience.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Compatible
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Effortlessly convert between Modules and pytrees using the Functional API for maximum
            flexibility.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Control
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Manage a Module's state with precision using typed Variable collections, enabling fine-grained
            control on JAX transformations.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: User-friendly
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            NNX prioritizes simplicity for common use cases, building upon lessons learned from Linen
            to provide a streamlined experience.

Basic usage
^^^^^^^^^^^^

.. testsetup::

   import jax
   import jax.numpy as jnp

.. testcode::

   from flax.experimental import nnx
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
NNX is under active development, we recommend using the latest version from Flax's GitHub repository:

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

      .. card:: :material-regular:`menu_book;2em` API reference
         :class-card: sd-text-black sd-bg-light
         :link: ../../api_reference/index.html


----

.. toctree::
   :hidden:
   :maxdepth: 1

   nnx_basics
   mnist_tutorial
   transforms