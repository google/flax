
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


Installation
^^^^^^^^^^^^
NNX is under active development, we recommend using the latest version from Flax's GitHub repository:

.. code-block:: bash

   pip install git+https://github.com/google/flax.git


Basic usage
^^^^^^^^^^^^

.. testsetup::

   import jax
   import jax.numpy as jnp

.. testcode::

   from flax.experimental import nnx

   class Linear(nnx.Module):
     def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
       key = rngs() # get a unique random key
       self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
       self.b = nnx.Param(jnp.zeros((dout,))) # initialize parameters
       self.din, self.dout = din, dout

     def __call__(self, x: jax.Array):
       return x @ self.w.value + self.b.value

   rngs = nnx.Rngs(0) # explicit RNG handling
   model = Linear(din=2, dout=3, rngs=rngs) # initialize the model

   x = jnp.empty((1, 2)) # generate random data
   y = model(x) # forward pass

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


----

.. toctree::
   :hidden:
   :maxdepth: 1

   nnx_basics
   mnist_tutorial