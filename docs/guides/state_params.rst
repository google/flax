.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/google/flax/blob/main/docs/notebooks/state_params.ipynb

Managing Parameters and State
=============================

We will show you how to...

* manage the variables from initialization to updates.
* split and re-assemble parameters and state.
* use :code:`vmap` with batch-dependant state.

.. testsetup::

  from flax import linen as nn
  from jax import random
  import jax.numpy as jnp
  import jax
  import optax

  # Create some fake data and run only for one epoch for testing.
  dummy_input = jnp.ones((3, 4))
  num_epochs = 1

.. testcode::

  class BiasAdderWithRunningMean(nn.Module):
    momentum: float = 0.9

    @nn.compact
    def __call__(self, x):
      is_initialized = self.has_variable('batch_stats', 'mean')
      mean = self.variable('batch_stats', 'mean', jnp.zeros, x.shape[1:])
      bias = self.param('bias', lambda rng, shape: jnp.zeros(shape), x.shape[1:])
      if is_initialized:
        mean.value = (self.momentum * mean.value +
                      (1.0 - self.momentum) * jnp.mean(x, axis=0, keepdims=True))
      return mean.value + bias

This example model is a minimal example that contains both parameters (declared
with :code:`self.param`) and state variables (declared with
:code:`self.variable`).

The tricky part with initialization here is that we need to split the state
variables and the parameters we're going to optimize for.

First we define ``update_step`` as follows (with a dummy loss that should be
replaced with yours):

.. testcode::

  def update_step(apply_fn, x, opt_state, params, state):
    def loss(params):
      y, updated_state = apply_fn({'params': params, **state},
                                  x, mutable=list(state.keys()))
      l = ((x - y) ** 2).sum() # Replace with your loss here.
      return l, updated_state

    (l, updated_state), grads = jax.value_and_grad(
        loss, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state)  # Defined below.
    params = optax.apply_updates(params, updates)
    return opt_state, params, updated_state

Then we can write the actual training code.

.. testcode::

  model = BiasAdderWithRunningMean()
  variables = model.init(random.PRNGKey(0), dummy_input)
  # Split state and params (which are updated by optimizer).
  state, params = variables.pop('params')
  del variables  # Delete variables to avoid wasting resources
  tx = optax.sgd(learning_rate=0.02)
  opt_state = tx.init(params)

  for _ in range(num_epochs):
    opt_state, params, state = update_step(
        model.apply, dummy_input, opt_state, params, state)


:code:`vmap` accross the batch dimension
----------------------------------------
When using :code:`vmap` and managing state that depends on the batch dimension,
for example when using :code:`BatchNorm`,  the setup above must be modified
slightly. This is because any layer whose state depends on the batch dimension
is not strictly vectorizable. In the case of :code:`BatchNorm`,
:code:`lax.pmean()` must be used to average the statistics over the batch
dimension so that the state is in sync for each item in the batch.

This requires two small changes. Firstly, we need to name the batch axis in our
model definition. Here, this is done by specifying the :code:`axis_name`
argument of :code:`BatchNorm`. In your own code this might require specifying
the :code:`axis_name` argument of :code:`lax.pmean()` directly.

.. testsetup::

  from functools import partial
  from flax import linen as nn
  from jax import random
  import jax.numpy as jnp
  import jax
  import optax

  # Create some fake data and run only for one epoch for testing.
  dummy_input = jnp.ones((100,))
  key1, key2 = random.split(random.PRNGKey(0), num=2)
  batch_size = 64
  X = random.normal(key1, (batch_size, 100))
  Y = random.normal(key2, (batch_size, 1))
  num_epochs = 1

.. testcode::

  class MLP(nn.Module):
    hidden_size: int
    out_size: int

    @nn.compact
    def __call__(self, x, train=False):
      norm = partial(
          nn.BatchNorm,
          use_running_average=not train,
          momentum=0.9,
          epsilon=1e-5,
          axis_name="batch", # Name batch dim
      )

      x = nn.Dense(self.hidden_size)(x)
      x = norm()(x)
      x = nn.relu(x)
      x = nn.Dense(self.hidden_size)(x)
      x = norm()(x)
      x = nn.relu(x)
      y = nn.Dense(self.out_size)(x)

      return y

Secondly, we need to specify the same name when calling :code:`vmap` in our training code:

.. testcode::

  def update_step(apply_fn, x_batch, y_batch, opt_state, params, state):

    def batch_loss(params):
      def loss_fn(x, y):
        pred, updated_state = apply_fn(
          {'params': params, **state},
          x, mutable=list(state.keys())
        )
        return (pred - y) ** 2, updated_state

      loss, updated_state = jax.vmap(
        loss_fn, out_axes=(0, None),  # Do not vmap `updated_state`.
        axis_name='batch'  # Name batch dim
      )(x_batch, y_batch)  # vmap only `x`, `y`, but not `state`.
      return jnp.mean(loss), updated_state

    (loss, updated_state), grads = jax.value_and_grad(
      batch_loss, has_aux=True
    )(params)

    updates, opt_state = tx.update(grads, opt_state)  # Defined below.
    params = optax.apply_updates(params, updates)
    return opt_state, params, updated_state, loss

Note that we also need to specify that the model state does not have a batch
dimension. Now we are able to train the model:

.. testcode::

  model = MLP(hidden_size=10, out_size=1)
  variables = model.init(random.PRNGKey(0), dummy_input)
  # Split state and params (which are updated by optimizer).
  state, params = variables.pop('params')
  del variables  # Delete variables to avoid wasting resources
  tx = optax.sgd(learning_rate=0.02)
  opt_state = tx.init(params)

  for _ in range(num_epochs):
    opt_state, params, state, loss = update_step(
        model.apply, X, Y, opt_state, params, state)
