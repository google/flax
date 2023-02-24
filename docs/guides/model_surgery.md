---
jupyter:
  jupytext:
    formats: md,ipynb
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
---

Model surgery
==============================

Usually, Flax modules and optimizers track and update the params for you. But there may be some time when you want to do some model surgery and tweak the param tensors yourself. This guide shows you how to do the trick.


## Setup

```python tags=["skip-execution"]
!pip install --upgrade -q pip jax jaxlib flax
```

```python
import functools

import jax
import jax.numpy as jnp
from flax import traverse_util
from flax import linen as nn
from flax.core import freeze
import jax
import optax
```

Surgery with Flax Modules
--------------------------------

Let's create a small convolutional neural network model for our demo.

As usual, you can run `CNN.init(...)['params']` to get the `params` to pass and modify it in every step of your training.


```python
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

def get_initial_params(key):
    init_shape = jnp.ones((1, 28, 28, 1), jnp.float32)
    initial_params = CNN().init(key, init_shape)['params']
    return initial_params

key = jax.random.PRNGKey(0)
params = get_initial_params(key)

jax.tree_util.tree_map(jnp.shape, params)
```

Note that what returned as `params` is a `FrozenDict`, which contains a few JAX arrays as kernel and bias. 

A `FrozenDict` is nothing more than a read-only dict, and Flax made it read-only because of the functional nature of JAX: JAX arrays are immutable, and the new `params` need to replace the old `params`. Making the dict read-only ensures that no in-place mutation of the dict can happen accidentally during the training and updating.

One way to actually modify the params outside of a Flax module is to explicitly flatten it and creates a mutable dict. Note that you can use a separator `sep` to join all nested keys. If no `sep` is given, the key will be a tuple of all nested keys.

```python
# Get a flattened key-value list.
flat_params = traverse_util.flatten_dict(params, sep='/')

jax.tree_util.tree_map(jnp.shape, flat_params)
```

Now you can do whatever you want with the params. When you are done, unflatten it back and use it in future training.

```python
# Somehow modify a layer
dense_kernel = flat_params['Dense_1/kernel']
flat_params['Dense_1/kernel'] = dense_kernel / jnp.linalg.norm(dense_kernel)

# Unflatten.
unflat_params = traverse_util.unflatten_dict(flat_params, sep='/')
# Refreeze.
unflat_params = freeze(unflat_params)
jax.tree_util.tree_map(jnp.shape, unflat_params)
```

Surgery with Optimizers
--------------------------------

When using `Optax` as an optimizer, the ``opt_state`` is actually a nested tuple
of the states of individual gradient transformations that compose the optimizer.
These states contain pytrees that mirror the parameter tree, and can be modified
the same way: flattening, modifying, unflattening, and then recreating a new
optimizer state that mirrors the original state.

```python
tx = optax.adam(1.0)
opt_state = tx.init(params)

# The optimizer state is a tuple of gradient transformation states.
jax.tree_util.tree_map(jnp.shape, opt_state)
```

The pytrees inside the optimizer state follow the same structure as the
parameters and can be flattened / modified exactly the same way.

```python
flat_mu = traverse_util.flatten_dict(opt_state[0].mu, sep='/')
flat_nu = traverse_util.flatten_dict(opt_state[0].nu, sep='/')

jax.tree_util.tree_map(jnp.shape, flat_mu)
```

After modification, re-create optimizer state. Use this for future training.

```python
opt_state = (
    opt_state[0]._replace(
        mu=traverse_util.unflatten_dict(flat_mu, sep='/'),
        nu=traverse_util.unflatten_dict(flat_nu, sep='/'),
    ),
) + opt_state[1:]
jax.tree_util.tree_map(jnp.shape, opt_state)
```
