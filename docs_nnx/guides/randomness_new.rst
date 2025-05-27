
Randomness
##########

.. testsetup:: Explicit, Implicit

  import jax
  import jax.numpy as jnp
  from flax import nnx


.. code-block:: python

  class RngStream:
    key: RngKey     # RngState < Variable
    count: RngCount # RngState < Variable
    tag: str

.. testcode:: Explicit

  from flax import nnx

  stream = nnx.RngStream(key=0, tag='dropout')

  key1 = stream()
  key2 = stream()


.. testcode:: Explicit

  rngs = nnx.Rngs(params=0, dropout=1)

  params_key = rngs.params()
  dropout_key = rngs.dropout()

.. codediff::
  :title: Explicit, Implicit
  :sync:

  class NoisyLinear(nnx.Module):
    __data__ = ('w', 'b')

    def __init__(self, din, dout, rngs: nnx.Rngs):
      self.w = nnx.Param(jax.random.normal(rngs.params(), (din, dout)))
      self.b = nnx.Param(jnp.zeros((dout,)))


    def __call__(self, x, rngs: nnx.Rngs):
      y = x @ self.w[...] + self.b[None]
      return y + 0.1 * jax.random.normal(rngs.noise(), x.shape)

  rngs = nnx.Rngs(params=0, noise=1)
  linear = NoisyLinear(10, 1, rngs=rngs)
  x = jax.random.normal(rngs.params(), (4, 10))
  y = linear(x, rngs=rngs)

  ---

  class NoisyLinear(nnx.Module):
    __data__ = ('w', 'b', 'noise')

    def __init__(self, din, dout, rngs: nnx.Rngs):
      self.w = nnx.Param(jax.random.normal(rngs.params(), (din, dout)))
      self.b = nnx.Param(jnp.zeros((dout,)))
      self.noise = rngs.noise.fork()  # get unique copy

    def __call__(self, x):
      y = x @ self.w[...] + self.b[None]
      return y + 0.1 * jax.random.normal(self.noise(), x.shape)

  rngs = nnx.Rngs(params=0, noise=1)
  linear = NoisyLinear(10, 1, rngs=rngs)
  x = jax.random.normal(rngs.params(), (4, 10))
  y = linear(x)

Default Stream
==============

.. testcode:: Explicit

  rngs = nnx.Rngs(0, params=1)

  key1 = rngs.default()       # uses 'default'
  key2 = rngs()               # uses 'default'
  key3 = rngs.params()        # uses 'params'
  key4 = rngs.dropout()       # uses 'default'
  key5 = rngs.unkown_stream() # uses 'default'

Standard Stream names
=====================

There are only two standard PRNG key stream names used by Flax NNX's built-in layers, shown in the table below:

- ``params``: used by most of the standard layers (e.g. ``Linear``, ``Conv``, ``MultiHeadAttention``, etc)
  during the construction to initialize their parameters.
- ``dropout``: used by ``nnx.Dropout`` and ``nnx.MultiHeadAttention`` to generate dropout masks.
- ``carry``: use by the recurrent layers (e.g. ``LSTMCell``, ``GRUCell``) to create the initial carry state.

Below is a simple example of a model that uses ``params`` and ``dropout`` PRNG key streams:


Filtering Random State
======================

.. testcode:: Explicit

  class Model(nnx.Module):
    __data__ = ('linear', 'dropout')
    def __init__(self, rngs: nnx.Rngs):
      self.linear = nnx.Linear(20, 10, rngs=rngs)
      self.dropout = nnx.Dropout(0.1, rngs=rngs)

  model = Model(nnx.Rngs(params=0, dropout=1))

  nnx.state(model, nnx.RngState) # All random states.
  nnx.state(model, nnx.RngKey) # Only PRNG keys.
  nnx.state(model, nnx.RngCount) # Only counts.
  nnx.state(model, 'params') # Only 'params'.
  nnx.state(model, 'dropout') # Only 'dropout'.
  nnx.state(model, nnx.All('params', nnx.RngKey)) # 'params' keys.