Batch normalization
===================

In this guide, you will learn how to apply `batch normalization <https://arxiv.org/abs/1502.03167>`__
using :meth:`flax.linen.BatchNorm <flax.linen.BatchNorm>`.

Batch normalization is a regularization technique used to speed up training and improve convergence.
During training, it computes running averages over feature dimensions. This adds a new form
of non-differentiable state that must be handled appropriately.

Throughout the guide, you will be able to compare code examples with and without Flax ``BatchNorm``.

.. testsetup::

  import flax.linen as nn
  import jax.numpy as jnp
  import jax
  import optax
  from typing import Any
  from flax.core import FrozenDict

Defining the model with ``BatchNorm``
*************************************

In Flax, ``BatchNorm`` is a :meth:`flax.linen.Module <flax.linen.Module>` that exhibits different runtime
behavior between training and inference. You explicitly specify it via the ``use_running_average`` argument,
as demonstrated below.

A common pattern is to accept a ``train`` (``training``) argument in the parent Flax ``Module``, and use
it to define ``BatchNorm``'s ``use_running_average`` argument.

Note: In other machine learning frameworks, like PyTorch or
TensorFlow (Keras), this is specified via a mutable state or a call flag (for example, in
`torch.nn.Module.eval <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval>`__
or ``tf.keras.Model`` by setting the
`training <https://www.tensorflow.org/api_docs/python/tf/keras/Model#call>`__ flag).

.. codediff::
  :title_left: No BatchNorm
  :title_right: With BatchNorm
  :sync:

  class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = nn.Dense(features=4)(x)

      x = nn.relu(x)
      x = nn.Dense(features=1)(x)
      return x

  ---
  class MLP(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool): #!
      x = nn.Dense(features=4)(x)
      x = nn.BatchNorm(use_running_average=not train)(x) #!
      x = nn.relu(x)
      x = nn.Dense(features=1)(x)
      return x

Once you create your model, initialize it by calling :meth:`flax.linen.init() <flax.linen.init>` to
get the ``variables`` structure. Here, the main difference between the code without ``BatchNorm``
and with ``BatchNorm`` is that the ``train`` argument must be provided.

The ``batch_stats`` collection
******************************

In addition to the ``params`` collection, ``BatchNorm`` also adds a ``batch_stats`` collection
that contains the running average of the batch statistics.

Note: You can learn more in the ``flax.linen`` `variables <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.core.variables>`__
API documentation.

The ``batch_stats`` collection must be extracted from the ``variables`` for later use.

.. codediff::
  :title_left: No BatchNorm
  :title_right: With BatchNorm
  :sync:

  mlp = MLP()
  x = jnp.ones((1, 3))
  variables = mlp.init(jax.random.PRNGKey(0), x)
  params = variables['params']


  jax.tree_util.tree_map(jnp.shape, variables)
  ---
  mlp = MLP()
  x = jnp.ones((1, 3))
  variables = mlp.init(jax.random.PRNGKey(0), x, train=False) #!
  params = variables['params']
  batch_stats = variables['batch_stats'] #!

  jax.tree_util.tree_map(jnp.shape, variables)


Flax ``BatchNorm`` adds a total of 4 variables: ``mean`` and ``var`` that live in the
``batch_stats`` collection, and ``scale`` and ``bias`` that live in the ``params``
collection.

.. codediff::
  :title_left: No BatchNorm
  :title_right: With BatchNorm
  :sync:

  FrozenDict({






    'params': {




      'Dense_0': {
          'bias': (4,),
          'kernel': (3, 4),
      },
      'Dense_1': {
          'bias': (1,),
          'kernel': (4, 1),
      },
    },
  })
  ---
  FrozenDict({
    'batch_stats': {     #!
      'BatchNorm_0': {   #!
          'mean': (4,),  #!
          'var': (4,),   #!
      },                 #!
    },                   #!
    'params': {
      'BatchNorm_0': {   #!
          'bias': (4,),  #!
          'scale': (4,), #!
      },                 #!
      'Dense_0': {
          'bias': (4,),
          'kernel': (3, 4),
      },
      'Dense_1': {
          'bias': (1,),
          'kernel': (4, 1),
      },
    },
  })

Modifying ``flax.linen.apply``
******************************

When using :meth:`flax.linen.apply <flax.linen.apply>` to run your model with the ``train==True``
argument (that is, you have ``use_running_average==False`` in the call to ``BatchNorm``), you
need to consider the following:

* ``batch_stats`` must be passed as an input variable.
* The ``batch_stats`` collection needs to be marked as mutable by setting ``mutable=['batch_stats']``.
* The mutated variables are returned as a second output.
  The updated ``batch_stats`` must be extracted from here.

.. codediff::
  :title_left: No BatchNorm
  :title_right: With BatchNorm
  :sync:

  y = mlp.apply(
    {'params': params},
    x,

  )
  ...

  ---
  y, updates = mlp.apply( #!
    {'params': params, 'batch_stats': batch_stats}, #!
    x,
    train=True, mutable=['batch_stats'] #!
  )
  batch_stats = updates['batch_stats'] #!

Training and evaluation
***********************

When integrating models that use ``BatchNorm`` into a training loop, the main challenge
is handling the additional ``batch_stats`` state. To do this, you need to:

* Add a ``batch_stats`` field to a custom :meth:`flax.training.train_state.TrainState <flax.training.train_state.TrainState>` class.
* Pass the ``batch_stats`` values to the :meth:`train_state.TrainState.create <train_state.TrainState.create>` method.

.. codediff::
  :title_left: No BatchNorm
  :title_right: With BatchNorm
  :sync:

  from flax.training import train_state




  state = train_state.TrainState.create(
    apply_fn=mlp.apply,
    params=params,

    tx=optax.adam(1e-3),
  )
  ---
  from flax.training import train_state

  class TrainState(train_state.TrainState):  #!
    batch_stats: Any  #!

  state = TrainState.create( #!
    apply_fn=mlp.apply,
    params=params,
    batch_stats=batch_stats, #!
    tx=optax.adam(1e-3),
  )

In addition, update your ``train_step`` function to reflect these changes:

* Pass all new parameters to ``flax.linen.apply`` (as previously discussed).
* The ``updates`` to the ``batch_stats`` must be propagated out of the ``loss_fn``.
* The ``batch_stats`` from the ``TrainState`` must be updated.

.. codediff::
  :title_left: No BatchNorm
  :title_right: With BatchNorm
  :sync:

  @jax.jit
  def train_step(state: TrainState, batch):
    """Train for a single step."""
    def loss_fn(params):
      logits = state.apply_fn(
        {'params': params},
        x=batch['image'])
      loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label'])
      return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    metrics = {
      'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
    }
    return state, metrics
  ---
  @jax.jit
  def train_step(state: TrainState, batch):
    """Train for a single step."""
    def loss_fn(params):
      logits, updates = state.apply_fn(  #!
        {'params': params, 'batch_stats': state.batch_stats},  #!
        x=batch['image'], train=True, mutable=['batch_stats']) #!
      loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label'])
      return loss, (logits, updates) #!
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params) #!
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats']) #!
    metrics = {
      'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
    }
    return state, metrics

The ``eval_step`` is much simpler. Because ``batch_stats`` is not mutable, no
updates
need to be propagated. Make sure you pass the ``batch_stats`` to ``flax.linen.apply``,
and the ``train`` argument is set to ``False``:

.. codediff::
  :title_left: No BatchNorm
  :title_right: With BatchNorm
  :sync:

  @jax.jit
  def eval_step(state: TrainState, batch):
    """Train for a single step."""
    logits = state.apply_fn(
      {'params': params},
      x=batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch['label'])
    metrics = {
      'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
    }
    return state, metrics
  ---
  @jax.jit
  def eval_step(state: TrainState, batch):
    """Train for a single step."""
    logits = state.apply_fn(
      {'params': params, 'batch_stats': state.batch_stats}, #!
      x=batch['image'], train=False) #!
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch['label'])
    metrics = {
      'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
    }
    return state, metrics
