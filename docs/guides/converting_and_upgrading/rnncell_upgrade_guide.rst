RNNCellBase Upgrade Guide
=========================

The ``RNNCellBase`` API has undergone some key updates aimed at enhancing usability:

- The ``initialize_carry`` method has transitioned from a class method to an instance method, simplifying its application.
- All necessary metadata is now stored directly within the cell instance, providing a streamlined method signature.

This guide will walk you through these changes, demonstrating how to update your existing code to align with these enhancements.

Basic Usage
-----------

.. testsetup:: New

  import flax.linen as nn
  import jax.numpy as jnp
  import jax
  import functools

Let's begin by defining some variables and a sample input that represents
a batch of sequences:

.. testcode:: New

  batch_size = 32
  seq_len = 10
  in_features = 64
  out_features = 128

  x = jnp.ones((batch_size, seq_len, in_features))

First and foremost, it's important to note that all metadata, including the number of features,
carry initializer, and so on, is now stored within the cell instance:

.. codediff::
  :title: Legacy, New
  :skip_test: Legacy
  :sync:

  cell = nn.LSTMCell()

  ---

  cell = nn.LSTMCell(features=out_features)

A significant change is that ``initialize_carry`` has been transitioned into an instance method. Given that
the cell instance now contains all metadata, the ``initialize_carry`` method's
signature only requires a PRNG key and a sample input:

.. codediff::
  :title: Legacy, New
  :skip_test: Legacy
  :sync:

  carry = nn.LSTMCell.initialize_carry(jax.random.key(0), (batch_size,), out_features)

  ---

  carry = cell.initialize_carry(jax.random.key(0), x[:, 0].shape)

Here, ``x[:, 0].shape`` represents the input for the cell (without the time dimension).
You can also just create the input shape directly when its more convenient:

.. testcode:: New

  carry = cell.initialize_carry(jax.random.key(0), (batch_size, in_features))


Upgrade Patterns
-----------------

The following sections will demonstrate some useful
patterns for updating your code to align with the new API.

First, we will show how to upgrade a ``Module`` that wraps
a cell, applies the scan logic during ``__call__``, and
has a static ``initialize_carry`` method. Here, we will try
to make the minimal amount of changes to the code to get
it working, albeit not in the most idiomatic way:

.. codediff::
  :title: Legacy, New
  :skip_test: Legacy
  :sync:

  class SimpleLSTM(nn.Module):

    @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):

      return nn.OptimizedLSTMCell()(carry, x)

    @staticmethod
    def initialize_carry(batch_dims, hidden_size):
      return nn.OptimizedLSTMCell.initialize_carry(
        jax.random.key(0), batch_dims, hidden_size)

  ---

  class SimpleLSTM(nn.Module):

    @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
      features = carry[0].shape[-1]
      return nn.OptimizedLSTMCell(features)(carry, x)

    @staticmethod
    def initialize_carry(batch_dims, hidden_size):
      return nn.OptimizedLSTMCell(hidden_size, parent=None).initialize_carry(
        jax.random.key(0), (*batch_dims, hidden_size))

Notice how in the new version, we have to extract the number of features from the carry
during ``__call__``, and use ``parent=None`` during ``initialize_carry`` to avoid some potential
side effects.

Next, we will show a more idiomatic way of writing a similar LSTM module. The main change
here will be that we will add a ``features`` attribute to the module and use it to initialize
a ``nn.scan``-ed version of the cell in the ``setup`` method:

.. codediff::
  :title: Legacy, New
  :skip_test: Legacy
  :sync:

  class SimpleLSTM(nn.Module):

    @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
      return nn.OptimizedLSTMCell()(carry, x)

    @staticmethod
    def initialize_carry(batch_dims, hidden_size):
      return nn.OptimizedLSTMCell.initialize_carry(
        jax.random.key(0), batch_dims, hidden_size)

  model = SimpleLSTM()
  carry = SimpleLSTM.initialize_carry((batch_size,), out_features)
  variables = model.init(jax.random.key(0), carry, x)

  ---

  class SimpleLSTM(nn.Module):
    features: int

    def setup(self):
      self.scan_cell = nn.transforms.scan(
        nn.OptimizedLSTMCell,
        variable_broadcast='params',
        in_axes=1, out_axes=1,
        split_rngs={'params': False})(self.features)


    @nn.compact
    def __call__(self, x):
      carry = self.scan_cell.initialize_carry(jax.random.key(0), x[:, 0].shape)
      return self.scan_cell(carry, x)[1]  # only return the output


  model = SimpleLSTM(features=out_features)
  variables = model.init(jax.random.key(0), x)

Because the ``carry`` can be easily initialized from the sample input, we can move the
call to ``initialize_carry`` into the ``__call__`` method, somewhat simplifying the code.

Development Notes
-----------------

When developing a new cell, consider the following:

* Include necessary metadata as instance attributes.
* The ``initialize_carry`` now only requires a PRNG key and a sample input.
* A new ``num_feature_axes`` property is required to specify the number of
  feature dimensions.

.. code-block::

  class LSTMCell(nn.RNNCellBase):
    features: int # ← All metadata is now stored within the cell instance
    ... #              ↓
    carry_init: Initializer

    def initialize_carry(self, rng, input_shape) -> Carry:
      ...

    @property
    def num_feature_axes(self):
      return 1

``num_feature_axes`` is a new API feature that allows code handling arbitrary ``RNNCellBase``
instances, such as the ``RNN`` Module, to infer the number of batch dimensions and
determine the position of the time axis.