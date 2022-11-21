Using Batch Normalization
===============

`Batch Normalization <https://arxiv.org/abs/1502.03167>`__ is a technique used to
speedup training and improve convergence, throught training it computes running averages over
the feature dimensions, this adds a new form of non-differentiable state that must be handled
appropriately. In this guide we will go through the details of using ``BatchNorm`` in models,
in the process we will highlight some of the differences between code that uses ``BatchNorm``
and code that does not.

.. testsetup::

  import flax.linen as nn
  import jax.numpy as jnp
  import jax
  import optax
  from typing import Any
  from flax.core import FrozenDict

Defining the model
******************

``BatchNorm`` is a Module that has different runtime behavior between training and
inference. In other frameworks this behavior is specified via mutable state or a call flag (e.g pytorch's `eval <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval>`__ method or Keras `training <https://www.tensorflow.org/api_docs/python/tf/keras/Model#call>`__ flag), however
in Flax it has to be explicitly specified via the ``use_running_average`` argument.
A common pattern is to accept a ``train`` argument in the parent Module and use it to define
``BatchNorm``'s ``use_running_average`` argument.

.. codediff::
  :title_left: regular code
  :title_right: with BatchNorm
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

Once the model is created, it can be initialized by calling ``init`` to get the ``variables`` structure.
The main difference is that the ``train`` argument is must be provided.

The ``batch_stats`` collection
******************************

Apart from the ``params`` collection, ``BatchNorm``
adds an additional ``batch_stats`` collection that contains the running
average of the batch statistics (for more info see the `variables <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.core.variables>`__ documentation). The ``batch_stats`` collection must be
extracted from the ``variables`` for later use:

.. codediff::
  :title_left: regular code
  :title_right: with BatchNorm
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


``BatchNorm`` adds a total of 4 variables: ``mean`` and ``var`` that live in the
``batch_stats`` collection and ``scale`` and ``bias`` that live in the ``params``
collection.

.. codediff::
  :title_left: regular code
  :title_right: with BatchNorm
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

Calling ``apply``
*************

When using ``apply`` to run your model with ``train==True``
(i.e., ``use_running_average==False` in the call to ``BatchNorm``),
a couple of things must be taken into consideration:

- ``batch_stats`` must be passed as an input variable.
- The ``batch_stats`` collection to be marked as
  mutable by setting ``mutable=['batch_stats']``.
- The mutated variables are returned as a second output.
  The updated ``batch_stats`` must be extracted from here.

.. codediff::
  :title_left: regular code
  :title_right: with BatchNorm
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

Training and Evaluation
***********************

When integrating models that use ``BatchNorm``into a training loop the main challenge
is to handle the additional ``batch_stats`` state. A way to do this is to add a
``batch_stats`` field to a custom ``TrainState`` class and passing the ``batch_stats``
values to the ``create`` method:

.. codediff::
  :title_left: regular code
  :title_right: with BatchNorm
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

Also the ``train_step`` function must be updated to reflect these changes, the main
differences are:

- All new parameters to ``apply`` must be passed (as discussed previously).
- The ``updates`` to the ``batch_stats`` must be propagated out of the ``loss_fn``.
- The ``batch_stats`` from the ``TrainState`` must be updated.

.. codediff::
  :title_left: regular code
  :title_right: with BatchNorm
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

The ``eval_step`` is much simpler, since ``batch_stats`` is not mutable no
updates need to be propagated. The only difference is that ``batch_stats`` must be
passed to ``apply``, and the ``train`` argument must be set to ``False``:

.. codediff::
  :title_left: regular code
  :title_right: with BatchNorm
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