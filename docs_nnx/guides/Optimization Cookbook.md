---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# A Flax Optimization Cookbook


# Exponential Moving Average

Neural network see increased robustness when, rather than using only the weights available at the end of training, we use an exponential moving average of the weights produced throughout training. It is easy to modify the standard Jax training loop to accomodate calculating exponential moving averages. 


## EMA in Pure Jax


To start, we will see how to keep track of exponential moving averages in raw Jax. Although the raw just implementation is simple and easy to understand, it does not allow for mutable state. 

```python
import jax.numpy as jnp
import jax
from jax import tree
import optax
import itertools as it
import functools as ft
from collections import namedtuple
```

```python
state = namedtuple('state', 'params opt_state ema_params')
```

```python
keys = map(ft.partial(jax.random.fold_in, jax.random.key(0)), it.count())
```

```python
x = jax.random.normal(next(keys), (32, 2))
y = jax.random.normal(next(keys), (32, 5))
```

```python
param_init = jax.nn.initializers.lecun_normal()
```

```python
def make_params(keys):
    return {
        'w': param_init(next(keys), (2, 5)),
        'b': jnp.zeros(5)
    }
```

```python
optimizer = optax.adam(1e-3)
```

```python
def make_state():
    params = make_params(keys)
    opt_state = optimizer.init(params)
    return state(params, opt_state, params)
```

```python
def ema_update(ema, new_val, decay=0.9):
    return decay * ema + (1 - decay) * new_val
```

```python
def model(params, x):
    return x @ params['w'] + params['b']
```

```python
def loss_fn(params, x, y):
    return jnp.sum((y - model(params, x))**2)
```

```python
@jax.jit
def train_step(x, y, st):
    loss, grads = jax.value_and_grad(loss_fn)(st.params, x, y)
    updates, opt_state = optimizer.update(grads, st.opt_state)
    params = optax.apply_updates(st.params, updates)
    ema_params = tree.map(ema_update, st.ema_params, params)
    return state(params, st.opt_state, ema_params), loss
```

```python
st = make_state()
```

```python
losses = []
for _ in range(50):
  st, loss = train_step(x, y, st)
  losses.append(loss)
```

## EMA in Flax


Now, we can see how to implement an exponential moving average in Flax. The code below is almost identical to the pure jax version above, but because NNX allows for mutable operations, we no longer need to explicitly pass around the full state object. 

```python
from flax import nnx
```

```python
nnx_model = nnx.Linear(2,5, rngs=nnx.Rngs(42))
```

```python
nnx_optimizer = nnx.Optimizer(
  nnx_model,
  tx=optimizer,
  wrt=nnx.Param,
)
```

```python
def nnx_loss_fn(model, x, y):
    return jnp.sum((model(x) - y) ** 2)
```

```python
class Ema(nnx.Module):
    def __init__(self, params):
        self.ema = nnx.merge(*nnx.split(nnx_model))
    def update(self, params):
        self.ema = tree.map(ema_update, nnx_ema, nnx_model)
```

```python
ema = Ema(nnx_model)
```

```python
@nnx.jit
def nnx_train_step(nnx_model, nnx_optimizer, ema, x, y):
  loss, grads = nnx.value_and_grad(nnx_loss_fn)(nnx_model, x, y)
  nnx_optimizer.update(nnx_model, grads)
  ema.update(nnx_model)
  return loss
```

```python
losses = []
for _ in range(50):
  loss = nnx_train_step(nnx_model, nnx_optimizer, ema, x, y)
  losses.append(loss)
```

# Low Rank Adaptation


The pattern for adding low rank adaptation to an optimization loop is very similar to adding an exponential moving average. As before, we create a new pytree with the same structure as our model parameters, but here we store low rank additions to these parameters rather than weighted average values. 


## Lora in Jax

```python
def init_lora_param(a, k=2):
    if len(a.shape) == 2:
        return {'A': param_init(next(keys), (a.shape[0], k)), 'B': jnp.zeros((k, a.shape[1]))}
    else:
        return None
```

```python
params = make_params(keys)
```

```python
lora_params = tree.map(init_lora_param, base_params)
```

```python
opt_state = optimizer.init(lora_params)
```

```python
def apply_lora_param(base_params, lora_params):
    if lora_params is None:
        return base_params
    return base_params + (lora_params['A'] @ lora_params['B'])
```

```python
def lora_loss(lora_params, params, x, y):
    params = tree.map(apply_lora_param, params, lora_params)
    return loss_fn(params, x, y)
```

```python
@jax.jit
def lora_train_step(x, y, params, lora_params, opt_state):
    loss, grads = jax.value_and_grad(lora_loss)(lora_params, params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    lora_params = optax.apply_updates(lora_params, updates)
    return params, lora_params, opt_state, loss
```

```python
losses = []
for _ in range(50):
  params, lora_params, opt_state, loss = lora_train_step(x, y, params, lora_params, opt_state)
  losses.append(loss)
```

## LORA in Flax


If Flax, we just need to wrap the optax optimizer with `nnx.Optimizer` to provide a mutable interface. 

```python
lora_params = tree.map(init_lora_param, nnx_model)
```

```python
def nnx_lora_loss(lora_params, params, x, y):
    params = tree.map(apply_lora_param, params, lora_params)
    return nnx_loss_fn(params, x, y)
```

```python
@nnx.jit
def nnx_lora_train_step(nnx_model, nnx_lora_params, nnx_optimizer, x, y):
  loss, grads = nnx.value_and_grad(nnx_lora_loss)(nnx_lora_params, nnx_model, x, y)
  nnx_optimizer.update(nnx_lora_params, grads)
  return loss
```

```python
nnx_optimizer = nnx.Optimizer(
  lora_params,
  tx=optimizer,
  wrt=nnx.Param,
)
```

```python
losses = []
for _ in range(50):
  loss = nnx_lora_train_step(nnx_model, lora_params, nnx_optimizer, x, y)
  losses.append(loss)
```

# LBFGS


## LBFGS in Jax

```python
def make_lbfgs_state(lbfgs):
    params = make_params(keys)
    opt_state = lbfgs.init(params)
    return (params, opt_state)
```

```python
@jax.jit
def train_step(x, y, params, opt_state):
    local_loss = lambda p: loss_fn(p, x, y)
    value_and_grad_fn = optax.value_and_grad_from_state(local_loss)
    loss, grad = value_and_grad_fn(params, state=opt_state)
    updates, opt_state = lbfgs.update(grad, opt_state, params,
                                      value=loss, grad=grad, value_fn=local_loss)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

```python
lbfgs = optax.lbfgs()
params, opt_state = make_lbfgs_state(lbfgs)
```

```python
losses = []
for _ in range(50):
  loss = train_step(x, y, params, opt_state)
  losses.append(loss)
```

## LBFGS in Flax


# TODO
- Per-param LR
- LBFGS
- Opt sharding different from variable sharding
- Gradient accumulation
