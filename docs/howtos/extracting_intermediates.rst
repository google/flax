Extracting intermediate values
==============================

This pattern will show you how to extract intermediate values from a module.
Let's start with this simple CNN that uses :code:`nn.compact`.

.. testsetup::

  import flax.linen as nn
  import jax
  import jax.numpy as jnp
  from flax.core import FrozenDict
  from typing import Sequence

  batch = jnp.ones((4, 32, 32, 3))

  class SowCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=32, kernel_size=(3, 3))(x)
      self.sow('intermediates', 'conv1', x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = nn.Conv(features=64, kernel_size=(3, 3))(x)
      self.sow('intermediates', 'conv2', x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = x.reshape((x.shape[0], -1))  # flatten
      self.sow('intermediates', 'features', x)
      x = nn.Dense(features=256)(x)
      self.sow('intermediates', 'conv3', x)
      x = nn.relu(x)
      x = nn.Dense(features=10)(x)
      self.sow('intermediates', 'dense', x)
      x = nn.log_softmax(x)
      return x

.. testcode::

  class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=32, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = nn.Conv(features=64, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = x.reshape((x.shape[0], -1))  # flatten
      x = nn.Dense(features=256)(x)
      x = nn.relu(x)
      x = nn.Dense(features=10)(x)
      x = nn.log_softmax(x)
      return x

Because this module uses ``nn.compact``, we don't have direct access to
intermediate values. There are a few ways to expose them:


Store intermediate values in a new variable collection
------------------------------------------------------

The CNN can be augmented with calls to ``sow`` to store intermediates as following:


.. codediff:: 
  :title_left: Default CNN
  :title_right: CNN using sow API
  
  class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=32, kernel_size=(3, 3))(x)

      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = nn.Conv(features=64, kernel_size=(3, 3))(x)

      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = x.reshape((x.shape[0], -1))  # flatten

      x = nn.Dense(features=256)(x)

      x = nn.relu(x)
      x = nn.Dense(features=10)(x)

      x = nn.log_softmax(x)
      return x
  ---
  class SowCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=32, kernel_size=(3, 3))(x)
      self.sow('intermediates', 'conv1', x) #!
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = nn.Conv(features=64, kernel_size=(3, 3))(x)
      self.sow('intermediates', 'conv2', x) #!
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = x.reshape((x.shape[0], -1))  # flatten
      self.sow('intermediates', 'features', x) #!
      x = nn.Dense(features=256)(x)
      self.sow('intermediates', 'conv3', x) #!
      x = nn.relu(x)
      x = nn.Dense(features=10)(x)
      self.sow('intermediates', 'dense', x) #!
      x = nn.log_softmax(x)
      return x

``sow`` only stores a value if the given variable collection is passed in
as "mutable" in the call to :code:`Module.apply`.

.. testcode::

  @jax.jit
  def init(key, x):
    variables = SowCNN().init(key, x)
    return variables

  @jax.jit
  def predict(variables, x):
    return SowCNN().apply(variables, x)

  @jax.jit
  def features(variables, x):
    # `mutable=['intermediates']` specified which collections are treated as
    # mutable during `apply`. The variables aren't actually mutated, instead
    # `apply` returns a second value, which is a dictionary of the modified
    # collections.
    output, modified_variables = SowCNN().apply(variables, x, mutable=['intermediates'])
    return modified_variables['intermediates']['features']

  variables = init(jax.random.PRNGKey(0), batch)
  predict(variables, batch)
  features(variables, batch)

Refactor module into submodules
-------------------------------

This is a useful pattern for cases where it's clear in what particular
way you want to split your submodules. Any submodule you expose in ``setup`` can be used directly. In the limit, you
can define all submodules in ``setup`` and avoid using ``nn.compact`` altogether.

.. testcode::

  class RefactoredCNN(nn.Module):
    def setup(self):
      self.features = Features()
      self.classifier = Classifier()

    def __call__(self, x):
      x = self.features(x)
      x = self.classifier(x)
      return x

  class Features(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=32, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = nn.Conv(features=64, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = x.reshape((x.shape[0], -1))  # flatten
      return x

  class Classifier(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = nn.Dense(features=256)(x)
      x = nn.relu(x)
      x = nn.Dense(features=10)(x)
      x = nn.log_softmax(x)
      return x

  @jax.jit
  def init(key, x):
    variables = RefactoredCNN().init(key, x)
    return variables['params']

  @jax.jit
  def features(params, x):
    return RefactoredCNN().apply({"params": params}, x,
      method=lambda module, x: module.features(x))

  params = init(jax.random.PRNGKey(0), batch)

  features(params, batch)


Use `capture_intermediates`
---------------------------

Linen supports the capture of intermediate return values from submodules automatically without any code changes.
This pattern should be considered the "sledge hammer" approach to capturing intermediates.
As a debugging and inspection tool it is very useful but using the other patterns described in this howto.

In the following code example we check if any intermediate activations are non-finite (NaN or infinite):

.. testcode::

  @jax.jit
  def init(key, x):
    variables = CNN().init(key, x)
    return variables

  @jax.jit
  def predict(variables, x):
    y, state = CNN().apply(variables, x, capture_intermediates=True, mutable=["intermediates"])
    intermediates = state['intermediates']
    fin = jax.tree_map(lambda xs: jnp.all(jnp.isfinite(xs)), intermediates)
    return y, fin

  variables = init(jax.random.PRNGKey(0), batch)
  y, is_finite = predict(variables, batch)
  all_finite = all(jax.tree_leaves(is_finite))
  assert all_finite, "non finite intermediate detected!"

By default only the intermediates of `__call__` methods are collected.
Alternatively, you can pass a custom filter based on the ``Module`` instance and the method name.

.. testcode::

  filter_Dense = lambda mdl, method_name: isinstance(mdl, nn.Dense)
  filter_encodings = lambda mdl, method_name: method_name == "encode"

  y, state = CNN().apply(variables, batch, capture_intermediates=filter_Dense, mutable=["intermediates"])
  dense_intermediates = state['intermediates']


Use ``nn.Sequential``
---------------------

You could also define ``CNN`` using a simple implementation of a ``Sequential`` combinator (this is quite common in more stateful approaches). This may be useful
for very simple models and gives you arbitrary model
surgery. But it can be very limiting -- if you even want to add one conditional, you are 
forced to refactor away from ``nn.Sequential`` and structure
your model more explicitly.

.. testcode::

  class Sequential(nn.Module):
    layers: Sequence[nn.Module]

    def __call__(self, x):
      for layer in self.layers:
        x = layer(x)
      return x

  def SeqCNN():
    return Sequential([
      nn.Conv(features=32, kernel_size=(3, 3)),
      nn.relu,
      lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
      nn.Conv(features=64, kernel_size=(3, 3)),
      nn.relu,
      lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
      lambda x: x.reshape((x.shape[0], -1)),  # flatten
      nn.Dense(features=256),
      nn.relu,
      nn.Dense(features=10),
      nn.log_softmax,
    ])

  @jax.jit
  def init(key, x):
    variables = SeqCNN().init(key, x)
    return variables['params']

  @jax.jit
  def features(params, x):
    return Sequential(SeqCNN().layers[0:7]).apply({"params": params}, x)

  params = init(jax.random.PRNGKey(0), batch)
  features(params, batch)
