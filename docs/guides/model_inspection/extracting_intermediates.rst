Extracting intermediate values
==============================

This guide will show you how to extract intermediate values from a module.
Let's start with this simple CNN that uses :code:`nn.compact`.

.. testsetup:: default, sow

  import flax
  import flax.linen as nn
  import jax
  import jax.numpy as jnp
  from flax.core import FrozenDict
  from typing import Sequence

  batch = jnp.ones((4, 32, 32, 3))

.. testcode::

  from flax import linen as nn
  import jax
  import jax.numpy as jnp
  from typing import Sequence

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
  :title: Default CNN, CNN using sow API
  :groups: default, sow

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
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = nn.Conv(features=64, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = x.reshape((x.shape[0], -1))  # flatten
      self.sow('intermediates', 'features', x) #!
      x = nn.Dense(features=256)(x)
      x = nn.relu(x)
      x = nn.Dense(features=10)(x)
      x = nn.log_softmax(x)
      return x

``sow`` acts as a no-op when the variable collection is not mutable.
Therefore, it works perfectly for debugging and optional tracking of intermediates.
The 'intermediates' collection is also used by the ``capture_intermediates`` API (see the :ref:`Use ``capture_intermediates``` section).

Note that, by default ``sow`` appends values every time it is called:

* This is necessary because once instantiated, a module could be called multiple
  times in its parent module, and we want to catch all the sowed values.
* Therefore you want to make sure that you **do not** feed intermediate values back
  into ``variables``. Otherwise every call will increase the length of that tuple
  and trigger a recompile.
* To override the default append behavior, specify ``init_fn`` and ``reduce_fn``
  - see :meth:`Module.sow() <flax.linen.Module.sow>`.

.. testcode:: sow

  class SowCNN2(nn.Module):
    @nn.compact
    def __call__(self, x):
      mod = SowCNN(name='SowCNN')
      return mod(x) + mod(x)  # Calling same module instance twice.

  @jax.jit
  def init(key, x):
    variables = SowCNN2().init(key, x)
    # By default the 'intermediates' collection is not mutable during init.
    # So variables will only contain 'params' here.
    return variables

  @jax.jit
  def predict(variables, x):
    # If mutable='intermediates' is not specified, then .sow() acts as a noop.
    output, mod_vars = SowCNN2().apply(variables, x, mutable='intermediates')
    features = mod_vars['intermediates']['SowCNN']['features']
    return output, features

  batch = jnp.ones((1,28,28,1))
  variables = init(jax.random.key(0), batch)
  preds, feats = predict(variables, batch)

  assert len(feats) == 2  # Tuple with two values since module was called twice.

Refactor module into submodules
-------------------------------

This is a useful pattern for cases where it's clear in what particular
way you want to split your submodules. Any submodule you expose in ``setup`` can
be used directly. In the limit, you can define all submodules in ``setup`` and
avoid using ``nn.compact`` altogether.

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

  params = init(jax.random.key(0), batch)

  features(params, batch)


Use ``capture_intermediates``
-----------------------------

Linen supports the capture of intermediate return values from submodules automatically without any code changes.
This pattern should be considered the "sledge hammer" approach to capturing intermediates.
As a debugging and inspection tool it is very useful, but using the other patterns described in this guide
will give you more fine-grained control over what intermediates you want to extract.

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
    fin = jax.tree_util.tree_map(lambda xs: jnp.all(jnp.isfinite(xs)), intermediates)
    return y, fin

  variables = init(jax.random.key(0), batch)
  y, is_finite = predict(variables, batch)
  all_finite = all(jax.tree_util.tree_leaves(is_finite))
  assert all_finite, "non-finite intermediate detected!"

By default only the intermediates of ``__call__`` methods are collected.
Alternatively, you can pass a custom filter function based on the ``Module`` instance and the method name.

.. testcode::

  filter_Dense = lambda mdl, method_name: isinstance(mdl, nn.Dense)
  filter_encodings = lambda mdl, method_name: method_name == "encode"

  y, state = CNN().apply(variables, batch, capture_intermediates=filter_Dense, mutable=["intermediates"])
  dense_intermediates = state['intermediates']

Note that ``capture_intermediates`` will only apply to layers. You can use ``self.sow`` to manually store
non-layer intermediates, but the filter function won't be applied to it.

.. codediff::
  :title: Capturing all layer intermediates, Using filter function and ``self.sow()``
  :groups: default, sow

  class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
      a = nn.Dense(4)(x) # Dense_0
      b = nn.Dense(4)(x) # Dense_1
      c = a + b # not a Flax layer, so won't be stored as an intermediate
      d = nn.Dense(4)(c) # Dense_2
      return d

  @jax.jit
  def init(key, x):
    variables = Model().init(key, x)
    return variables['params']

  @jax.jit
  def predict(params, x):
    return Model().apply({"params": params}, x, capture_intermediates=True)

  batch = jax.random.uniform(jax.random.key(1), (1,3))
  params = init(jax.random.key(0), batch)
  preds, feats = predict(params, batch)
  feats # intermediate c in Model was not stored because it's not a Flax layer
  ---
  class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
      a = nn.Dense(4)(x) # Dense_0
      b = nn.Dense(4)(x) # Dense_1
      c = a + b
      self.sow('intermediates', 'c', c) # store intermediate c #!
      d = nn.Dense(4)(c) # Dense_2
      return d

  @jax.jit
  def init(key, x):
    variables = Model().init(key, x)
    return variables['params']

  @jax.jit
  def predict(params, x):
    # filter specifically for only the Dense_0 and Dense_2 layer #!
    filter_fn = lambda mdl, method_name: isinstance(mdl.name, str) and (mdl.name in {'Dense_0', 'Dense_2'}) #!
    return Model().apply({"params": params}, x, capture_intermediates=filter_fn) #!

  batch = jax.random.uniform(jax.random.key(1), (1,3))
  params = init(jax.random.key(0), batch)
  preds, feats = predict(params, batch)
  feats # intermediate c in Model is stored and isn't filtered out by the filter function #!

To separate the intermediates extracted from ``self.sow`` from the intermediates extracted from ``capture_intermediates``,
we can either define a separate collection like ``self.sow('sow_intermediates', 'c', c)``, or manually filter out
the intermediates after calling ``.apply()``. For example:

.. testcode:: sow

  flattened_dict = flax.traverse_util.flatten_dict(feats['intermediates'], sep='/')
  flattened_dict['c']

In terms of efficiency, as long as everything is jitted, then any intermediates you don't end up using
should be optimized away by XLA.

Use ``Sequential``
---------------------

You could also define ``CNN`` using a simple implementation of a ``Sequential`` combinator (this is quite common in more stateful approaches). This may be useful
for very simple models and gives you arbitrary model
surgery. But it can be very limiting -- if you even want to add one conditional, you are
forced to refactor away from ``Sequential`` and structure
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

  batch = jnp.ones((1,28,28,1))
  params = init(jax.random.key(0), batch)
  features(params, batch)

Extracting gradients of intermediate values
===========================================
For debugging purposes, it can be useful to extract the gradients of intermediate values.
This can be done by using the :meth:`Module.perturb() <flax.linen.Module.perturb>` method over the desired values.

.. testcode::

  class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = nn.relu(nn.Dense(8)(x))
      x = self.perturb('hidden', x)
      x = nn.Dense(2)(x)
      x = self.perturb('logits', x)
      return x

``perturb`` adds a variable to a ``perturbations`` collection by default,
it behaves like an identity function and the gradient of the perturbation
matches the gradient of the input. To get the perturbations just initialize
the model:

.. testcode::

  x = jnp.empty((1, 4)) # random data
  y = jnp.empty((1, 2)) # random data

  model = Model()
  variables = model.init(jax.random.key(1), x)
  params, perturbations = variables['params'], variables['perturbations']

Finally compute the gradients of the loss with respect to the perturbations,
these will match the gradients of the intermediates:

.. testcode::

  def loss_fn(params, perturbations, x, y):
    y_pred = model.apply({'params': params, 'perturbations': perturbations}, x)
    return jnp.mean((y_pred - y) ** 2)

  intermediate_grads = jax.grad(loss_fn, argnums=1)(params, perturbations, x, y)