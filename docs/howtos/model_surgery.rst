Model Surgery
==============================

.. testsetup::

  import functools
  import numpy as np
  import jax
  from jax import lax, random, numpy as jnp
  import flax
  from flax import optim, traverse_util

  from flax import linen as nn
  from flax.core import unfreeze, freeze

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

  key = random.PRNGKey(0)
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

  # Unfreeze params to normal dict.
  params = unfreeze(params)
  # Get flattened-key: value list.
  flat_params = {'/'.join(k): v for k, v in traverse_util.flatten_dict(params).items()}
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
  unflat_params = traverse_util.unflatten_dict({tuple(k.split('/')): v for k, v in flat_params.items()})
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

Surgey with Optimizers
--------------------------------

If you're loading from a Flax optimizer, all of the variables live in
``optimizer.target``.

.. testcode::

  opt_def = optim.Adam(1.0)
  opt = opt_def.create(params)

  # Get optimizer state and target vars by:
  opt_state = opt.state_dict()
  print(jax.tree_map(jnp.shape, opt_state))

.. testoutput::
  :options: +NORMALIZE_WHITESPACE
  
  {'state': {'param_states': {'Conv_0': {'bias': {'grad_ema': (32,),
      'grad_sq_ema': (32,)},
      'kernel': {'grad_ema': (3, 3, 1, 32), 'grad_sq_ema': (3, 3, 1, 32)}},
    'Conv_1': {'bias': {'grad_ema': (64,), 'grad_sq_ema': (64,)},
      'kernel': {'grad_ema': (3, 3, 32, 64), 'grad_sq_ema': (3, 3, 32, 64)}},
    'Dense_0': {'bias': {'grad_ema': (256,), 'grad_sq_ema': (256,)},
      'kernel': {'grad_ema': (3136, 256), 'grad_sq_ema': (3136, 256)}},
    'Dense_1': {'bias': {'grad_ema': (10,), 'grad_sq_ema': (10,)},
      'kernel': {'grad_ema': (256, 10), 'grad_sq_ema': (256, 10)}}},
    'step': ()},
  'target': {'Conv_0': {'bias': (32,), 'kernel': (3, 3, 1, 32)},
    'Conv_1': {'bias': (64,), 'kernel': (3, 3, 32, 64)},
    'Dense_0': {'bias': (256,), 'kernel': (3136, 256)},
    'Dense_1': {'bias': (10,), 'kernel': (256, 10)}}}

.. testcode::

  # Get flattened-key:: value list.
  flat_opt_state = {'/'.join(k): v for k, v in traverse_util.flatten_dict(opt_state).items()}
  print(jax.tree_map(jnp.shape, flat_opt_state))

.. testoutput::
  :options: +NORMALIZE_WHITESPACE
  
  {'state/param_states/Conv_0/bias/grad_ema': (32,),
  'state/param_states/Conv_0/bias/grad_sq_ema': (32,),
  'state/param_states/Conv_0/kernel/grad_ema': (3, 3, 1, 32),
  'state/param_states/Conv_0/kernel/grad_sq_ema': (3, 3, 1, 32),
  'state/param_states/Conv_1/bias/grad_ema': (64,),
  'state/param_states/Conv_1/bias/grad_sq_ema': (64,),
  'state/param_states/Conv_1/kernel/grad_ema': (3, 3, 32, 64),
  'state/param_states/Conv_1/kernel/grad_sq_ema': (3, 3, 32, 64),
  'state/param_states/Dense_0/bias/grad_ema': (256,),
  'state/param_states/Dense_0/bias/grad_sq_ema': (256,),
  'state/param_states/Dense_0/kernel/grad_ema': (3136, 256),
  'state/param_states/Dense_0/kernel/grad_sq_ema': (3136, 256),
  'state/param_states/Dense_1/bias/grad_ema': (10,),
  'state/param_states/Dense_1/bias/grad_sq_ema': (10,),
  'state/param_states/Dense_1/kernel/grad_ema': (256, 10),
  'state/param_states/Dense_1/kernel/grad_sq_ema': (256, 10),
  'state/step': (),
  'target/Conv_0/bias': (32,),
  'target/Conv_0/kernel': (3, 3, 1, 32),
  'target/Conv_1/bias': (64,),
  'target/Conv_1/kernel': (3, 3, 32, 64),
  'target/Dense_0/bias': (256,),
  'target/Dense_0/kernel': (3136, 256),
  'target/Dense_1/bias': (10,),
  'target/Dense_1/kernel': (256, 10)}

.. testcode::

    # Unflatten
    unflat_opt_state = traverse_util.unflatten_dict({tuple(k.split('/')): v for k, v in flat_opt_state.items()})
    print(jax.tree_map(jnp.shape, unflat_opt_state))

.. testoutput::
  :options: +NORMALIZE_WHITESPACE
  
  {'state': {'param_states': {'Conv_0': {'bias': {'grad_ema': (32,),
      'grad_sq_ema': (32,)},
      'kernel': {'grad_ema': (3, 3, 1, 32), 'grad_sq_ema': (3, 3, 1, 32)}},
    'Conv_1': {'bias': {'grad_ema': (64,), 'grad_sq_ema': (64,)},
      'kernel': {'grad_ema': (3, 3, 32, 64), 'grad_sq_ema': (3, 3, 32, 64)}},
    'Dense_0': {'bias': {'grad_ema': (256,), 'grad_sq_ema': (256,)},
      'kernel': {'grad_ema': (3136, 256), 'grad_sq_ema': (3136, 256)}},
    'Dense_1': {'bias': {'grad_ema': (10,), 'grad_sq_ema': (10,)},
      'kernel': {'grad_ema': (256, 10), 'grad_sq_ema': (256, 10)}}},
    'step': ()},
  'target': {'Conv_0': {'bias': (32,), 'kernel': (3, 3, 1, 32)},
    'Conv_1': {'bias': (64,), 'kernel': (3, 3, 32, 64)},
    'Dense_0': {'bias': (256,), 'kernel': (3136, 256)},
    'Dense_1': {'bias': (10,), 'kernel': (256, 10)}}}

We can restore the optimizer object from the nested-dict state. The restored 
state must agree with the shape of the existing object as a sort of "structural
unit test".

.. testcode::

  restored_opt = opt.restore_state(unflat_opt_state)
  print(jax.tree_map(jnp.shape, restored_opt))

.. testoutput::
  :options: +NORMALIZE_WHITESPACE, +ELLIPSIS

  Optimizer(optimizer_def=<flax.optim.adam.Adam object at ...>, state=OptimizerState(step=(), param_states={'Conv_0': {'bias': _AdamParamState(grad_ema=(32,), grad_sq_ema=(32,)), 'kernel': _AdamParamState(grad_ema=(3, 3, 1, 32), grad_sq_ema=(3, 3, 1, 32))}, 'Conv_1': {'bias': _AdamParamState(grad_ema=(64,), grad_sq_ema=(64,)), 'kernel': _AdamParamState(grad_ema=(3, 3, 32, 64), grad_sq_ema=(3, 3, 32, 64))}, 'Dense_0': {'bias': _AdamParamState(grad_ema=(256,), grad_sq_ema=(256,)), 'kernel': _AdamParamState(grad_ema=(3136, 256), grad_sq_ema=(3136, 256))}, 'Dense_1': {'bias': _AdamParamState(grad_ema=(10,), grad_sq_ema=(10,)), 'kernel': _AdamParamState(grad_ema=(256, 10), grad_sq_ema=(256, 10))}}), target={'Conv_0': {'bias': (32,), 'kernel': (3, 3, 1, 32)}, 'Conv_1': {'bias': (64,), 'kernel': (3, 3, 32, 64)}, 'Dense_0': {'bias': (256,), 'kernel': (3136, 256)}, 'Dense_1': {'bias': (10,), 'kernel': (256, 10)}})
