Model Surgery
==============================

.. testsetup::

  import functools

  import jax
  import jax.numpy as jnp
  from flax import traverse_util
  from flax import linen as nn
  from flax.core import freeze
  import jax
  import optax

We will show how to get a flat dict of all the tensors, and then go back to a 
nested, frozen dict. This will be demonstrated for both Flax modules and optimizers.

Surgery with Flax Modules
--------------------------------

Let's create a small convolutional neural network model for our demo.

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
      x = x.reshape((x.shape[0], -1))
      x = nn.Dense(features=256)(x)
      x = nn.relu(x)
      x = nn.Dense(features=10)(x)
      x = nn.log_softmax(x)
      return x

  def get_initial_params(rng):
    init_shape = jnp.ones((1, 28, 28, 1), jnp.float32)
    initial_params = CNN().init(rng, init_shape)['params']
    return initial_params

  key = jax.random.PRNGKey(0)
  params = get_initial_params(key)

  print(jax.tree_map(jnp.shape, params))

.. testoutput::

  FrozenDict({
      Conv_0: {
          bias: (32,),
          kernel: (3, 3, 1, 32),
      },
      Conv_1: {
          bias: (64,),
          kernel: (3, 3, 32, 64),
      },
      Dense_0: {
          bias: (256,),
          kernel: (3136, 256),
      },
      Dense_1: {
          bias: (10,),
          kernel: (256, 10),
      },
  })


Next, get a flat dict for doing model surgery as follows:

.. testcode::

  # Get flattened-key: value list.
  flat_params = traverse_util.flatten_dict(params, sep='/')
  print(jax.tree_map(jnp.shape, flat_params))

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  {'Conv_0/bias': (32,),
   'Conv_0/kernel': (3, 3, 1, 32),
   'Conv_1/bias': (64,),
   'Conv_1/kernel': (3, 3, 32, 64),
   'Dense_0/bias': (256,),
   'Dense_0/kernel': (3136, 256),
   'Dense_1/bias': (10,),
   'Dense_1/kernel': (256, 10)}

After doing whatever you want, unflatten back:

.. testcode::

  # Unflatten.
  unflat_params = traverse_util.unflatten_dict(flat_params, sep='/')
  # Refreeze.
  unflat_params = freeze(unflat_params)
  print(jax.tree_map(jnp.shape, unflat_params))

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  FrozenDict({
      Conv_0: {
          bias: (32,),
          kernel: (3, 3, 1, 32),
      },
      Conv_1: {
          bias: (64,),
          kernel: (3, 3, 32, 64),
      },
      Dense_0: {
          bias: (256,),
          kernel: (3136, 256),
      },
      Dense_1: {
          bias: (10,),
          kernel: (256, 10),
      },
  })

Surgery with Optimizers
--------------------------------

When using `Optax` as an optimizer, the ``opt_state`` is actually a nested tuple
of the states of individual gradient transformations that compose the optimizer.
These states contain pytrees that mirror the parameter tree, and can be modified
the same way: flattening, modifying, unflattening, and then recreating a new
optimizer state that mirrors the original state.

.. testcode::

  tx = optax.adam(1.0)
  opt_state = tx.init(params)

  # The optimizer state is a tuple of gradient transformation states.
  print(jax.tree_map(jnp.shape, opt_state))

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  (ScaleByAdamState(count=(), mu=FrozenDict({
      Conv_0: { bias: (32,), kernel: (3, 3, 1, 32), },
      Conv_1: { bias: (64,), kernel: (3, 3, 32, 64), },
      Dense_0: { bias: (256,), kernel: (3136, 256), },
      Dense_1: { bias: (10,), kernel: (256, 10), },
  }), nu=FrozenDict({
      Conv_0: { bias: (32,), kernel: (3, 3, 1, 32), },
      Conv_1: { bias: (64,), kernel: (3, 3, 32, 64), },
      Dense_0: { bias: (256,), kernel: (3136, 256), },
      Dense_1: { bias: (10,), kernel: (256, 10), },
  })), EmptyState())
  
The pytrees inside the optimizer state follow the same structure as the
parameters and can be flattened / modified exactly the same way

.. testcode::

  flat_mu = traverse_util.flatten_dict(opt_state[0].mu, sep='/')
  flat_nu = traverse_util.flatten_dict(opt_state[0].nu, sep='/')

  print(jax.tree_map(jnp.shape, flat_mu))

.. testoutput::
  :options: +NORMALIZE_WHITESPACE
  
  {'Conv_0/bias': (32,),
   'Conv_0/kernel': (3, 3, 1, 32),
   'Conv_1/bias': (64,),
   'Conv_1/kernel': (3, 3, 32, 64),
   'Dense_0/bias': (256,),
   'Dense_0/kernel': (3136, 256),
   'Dense_1/bias': (10,),
   'Dense_1/kernel': (256, 10)}

After modification, re-create optimizer state:

.. testcode::

  opt_state = (
      opt_state[0]._replace(
          mu=traverse_util.unflatten_dict(flat_mu, sep='/'),
          nu=traverse_util.unflatten_dict(flat_nu, sep='/'),
      ),
  ) + opt_state[1:]
