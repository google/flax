.. Flax documentation main file, created by
   sphinx-quickstart on Mon Feb 17 11:41:38 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************************
Flax
******************************


.. div:: sd-text-left sd-font-italic

   Neural networks with JAX


----

Flax delivers an **end-to-end, flexible, user experience for researchers
who use JAX with neural networks**. Flax exposes the full power of JAX.
It is made up of loosely coupled libraries,
which are showcased with end-to-end integrated guides and examples.


Features
^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Safety
         :class-card: sd-text-black sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Flax is designed for correctness and safety. Thanks to its immutable Modules
            and Functional API, Flax helps mitigate bugs that araise when handling state
            in JAX.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Control
         :class-card: sd-text-black sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Flax grants more fine grained control and expressivity than most Neural Network
            frameworks via its Variable Collections, RNG Collections and Mutability conditions.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Functional API
         :class-card: sd-text-black sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Flax's functional API radically redefines what Modules can do via lifted transformations like vmap, scan, etc, while also enabling seamless integration with other JAX libraries like Optax and Chex.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Terse Code
         :class-card: sd-text-black sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Flax's :meth:`compact <flax.linen.compact>` Modules enables submodules to be defined directly at their callsite, leading to code that is easier to read and avoids repetition.


----

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install flax

Flax installs the vanilla CPU version of JAX, if you need a custom version please check out `JAX's installation page <https://github.com/google/jax#installation>`__.

Basic usage
^^^^^^^^^^^^

.. testsetup::

   import jax
   from jax.random import PRNGKey
   import flax.linen as nn
   import jax.numpy as jnp

.. testcode::

   class MLP(nn.Module):                    # create a Flax Module dataclass
     out_dims: int

     @nn.compact
     def __call__(self, x):
       x = x.reshape((x.shape[0], -1))
       x = nn.Dense(128)(x)                 # create inline Flax Module submodules
       x = nn.relu(x)
       x = nn.Dense(self.out_dims)(x)       # shape inference
       return x

   model = MLP(out_dims=10)                 # instantiate the MLP model

   x = jnp.empty((4, 28, 28, 1))            # generate random data
   variables = model.init(PRNGKey(42), x)   # initialize the weights
   y = model.apply(variables, x)            # make forward pass

----

Learn more
^^^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` Getting Started
         :class-card: sd-text-black sd-bg-light
         :link: getting_started.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` Guides
         :class-card: sd-text-black sd-bg-light
         :link: guides/index.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`settings;2em` Advanced Topics
         :class-card: sd-text-black sd-bg-light
         :link: advanced_topics/index.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`science;2em` Examples
         :class-card: sd-text-black sd-bg-light
         :link: examples.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`menu_book;2em` API Reference
         :class-card: sd-text-black sd-bg-light
         :link: api_reference/index.html

----

Ecosystem
^^^^^^^^^

Flax is used by `hundreds of projects (and growing) <https://github.com/google/flax/network/dependents?package_id=UGFja2FnZS01MjEyMjA2MA%3D%3D>`__,
both in the open source community and within Google.
Notable examples include:


.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: `🤗 Hugging Face <https://huggingface.co/flax-community>`__
         :class-card: sd-text-black sd-border-0
         :shadow: none
         :class-title: sd-text-center sd-fs-5

         .. div:: sd-text-center sd-font-italic

            NLP and Computer Vision models

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: `🥑 DALLE Mini <https://huggingface.co/dalle-mini>`__
         :class-card: sd-text-black sd-border-0
         :shadow: none
         :class-title: sd-text-center sd-fs-5

         .. div:: sd-text-center sd-font-italic

            Model for Text-to-Image generation

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: `PaLM <https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html>`__
         :class-card: sd-text-black sd-border-0
         :shadow: none
         :class-title: sd-text-center sd-fs-5

         .. div:: sd-text-center sd-font-italic

            540 Billion parameter model for text generation

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: `Imagen <https://imagen.research.google>`__
         :class-card: sd-text-black sd-border-0
         :shadow: none
         :class-title: sd-text-center sd-fs-5

         .. div:: sd-text-center sd-font-italic

            Text-to-Image Diffusion Models

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: `Big Vision <https://github.com/google-research/big_vision>`__
         :class-card: sd-text-black sd-border-0
         :shadow: none
         :class-title: sd-text-center sd-fs-5

         .. div:: sd-text-center sd-font-italic

            Large scale Computer Vision models

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: `T5x <https://github.com/google-research/t5x>`__
         :class-card: sd-text-black sd-border-0
         :shadow: none
         :class-title: sd-text-center sd-fs-5

         .. div:: sd-text-center sd-font-italic

            Large Language Models

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: `Brax <https://github.com/google/brax>`__
         :class-card: sd-text-black sd-border-0
         :shadow: none
         :class-title: sd-text-center sd-fs-5

         .. div:: sd-text-center sd-font-italic

            On-device differentiable RL environments




.. toctree::
   :hidden:
   :maxdepth: 2

   Getting Started <getting_started>
   guides/index
   examples
   advanced_topics/index
   🔪 Flax - The Sharp Bits 🔪 <notebooks/flax_sharp_bits>
   contributing/index
   api_reference/index
