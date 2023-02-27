---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

+++ {"id": "dHMnJTK9R5n9"}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/flax/blob/main/docs/notebooks/optax_update_guide.ipynb)
[![Open On GitHub](https://img.shields.io/badge/Open-on%20GitHub-blue?logo=GitHub)](https://github.com/google/flax/blob/main/docs/notebooks/optax_update_guide.ipynb)

Colab for
https://flax.readthedocs.io/en/latest/guides/optax_update_guide.html

+++ {"id": "fCCY-S009eHv"}

### Setup

```{code-cell}
:id: I4PiwrnnO6Fw
:tags: [skip-execution]

# flax.optim was deprecated after 0.5.3
!pip install -q --force-reinstall flax==0.5.3 optax
```

```{code-cell}
:id: 7hDWlLOOt4U6

from typing import Sequence

import flax
from  flax.training import train_state
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.optim
import optax
```

```{code-cell}
:id: mb2xGRAwueSa
:outputId: 30260605-6773-482d-be0e-0383e63b9fa2

batch = {
    'image': jnp.ones([1, 28, 28, 1]),
    'label': jnp.array([0]),
}
```

```{code-cell}
:id: lf11Nzj-t32w
:outputId: c54a570b-d76a-43bb-f1ab-5cbbbfd6f584

class Perceptron(nn.Module):
  units: Sequence[int]
  @nn.compact
  def __call__(self, x):
    x = x.reshape([x.shape[0], -1]) / 255.
    x = nn.Dense(50)(x)
    x = nn.relu(x)
    return nn.Dense(10)(x)

def loss(params, batch):
  logits = model.apply({'params': params}, batch['image'])
  one_hot = jax.nn.one_hot(batch['label'], 10)
  return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))

model = Perceptron([50, 10])
variables = model.init(jax.random.PRNGKey(0), batch['image'])

jax.tree_util.tree_map(jnp.shape, variables)
```

```{code-cell}
:id: xKm0nn4X57Vg
:outputId: e1794ea2-ea91-45b5-8b89-d39d2d923cc5

import tensorflow_datasets as tfds

builder = tfds.builder('mnist')
builder.download_and_prepare()
ds_test = jax.tree_util.tree_map(jnp.array, builder.as_dataset('test', batch_size=-1))
get_ds_train = lambda: (
    jax.tree_util.tree_map(jnp.array, x)
    for x in builder.as_dataset('train').batch(128))
batch = next(get_ds_train())
jax.tree_util.tree_map(jnp.shape, batch)
```

```{code-cell}
:id: eCceGZ_Kvko5
:outputId: af1f8657-b3eb-4c9b-d073-48954481166b

@jax.jit
def eval(params):
  logits = model.apply({'params': params}, ds_test['image'])
  return (logits.argmax(axis=-1) == ds_test['label']).mean()

eval(variables['params'])
```

```{code-cell}
:id: rqQiq3ugxKjX

learning_rate, momentum = 0.01, 0.9
```

+++ {"id": "JyXnY2WW9gfR"}

### Replacing `flax.optim` with `optax`

```{code-cell}
:id: 7mlz-P5AwBc2
:outputId: 05cb4271-e407-4798-d0ea-cd8dfbd9f2a1

@jax.jit
def train_step(optimizer, batch):
  grads = jax.grad(loss)(optimizer.target, batch)
  return optimizer.apply_gradient(grads)

optimizer = flax.optim.Momentum(learning_rate, momentum).create(
    variables['params'])
for batch in get_ds_train():
  optimizer = train_step(optimizer, batch)

eval(optimizer.target)
```

```{code-cell}
:id: n9IxglwJxD3X
:outputId: a42f9e8d-d1fe-4221-ec28-b7258174b16c

tx = optax.sgd(learning_rate, momentum)
params = variables['params']
opt_state = tx.init(params)

@jax.jit
def train_step(params, opt_state, batch):
  grads = jax.grad(loss)(params, batch)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state

for batch in get_ds_train():
  params, opt_state = train_step(params, opt_state, batch)

eval(params)
```

```{code-cell}
:id: mBcp17BEvBVs
:outputId: dd912efc-669f-42c2-86b2-e242cecc67ce

@jax.jit
def train_step(state, batch):
  def loss(params):
    logits = state.apply_fn({'params': params}, batch['image'])
    one_hot = jax.nn.one_hot(batch['label'], 10)
    return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
  grads = jax.grad(loss)(state.params)
  return state.apply_gradients(grads=grads)

tx = optax.sgd(learning_rate, momentum)
state = train_state.TrainState.create(
    apply_fn=model.apply, tx=tx, params=variables['params'],
)
opt_state = tx.init(params)

for batch in get_ds_train():
  state = train_step(state, batch)

eval(params)
```

+++ {"id": "4ute1zBpRnaq"}

### Composable Gradient Transformations

```{code-cell}
:id: M2WjJ7HT8GMn
:outputId: 7ae7117e-5d1d-44ca-abb8-65b0db2c0eb6

@jax.jit
def train_step(params, opt_state, batch):
  grads = jax.grad(loss)(params, batch)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state

tx = optax.chain(
    optax.trace(decay=momentum),
    optax.scale(-learning_rate),
)
params = variables['params']
opt_state = tx.init(params)

for batch in get_ds_train():
  params, opt_state = train_step(params, opt_state, batch)

eval(params)
```

+++ {"id": "vFt96TU-rSQM"}

### Weight Decay

```{code-cell}
:id: Qbq9vK24-omQ

weight_decay = 1e-5
```

```{code-cell}
:id: cx1YCFVL9ktA
:outputId: bb6795a7-c5c2-458a-d3e9-4b769b857fed

@jax.jit
def train_step(optimizer, batch):
  grads = jax.grad(loss)(optimizer.target, batch)
  return optimizer.apply_gradient(grads)

optimizer = flax.optim.Adam(learning_rate, weight_decay=weight_decay).create(
    variables['params'])
for batch in get_ds_train():
  optimizer = train_step(optimizer, batch)

eval(optimizer.target)
```

```{code-cell}
:id: kIWCQz33-p98
:outputId: 84753c79-314b-4982-97e6-0c61a490eab1

@jax.jit
def train_step(params, opt_state, batch):
  grads = jax.grad(loss)(params, batch)
  updates, opt_state = tx.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state

tx = optax.chain(
    optax.scale_by_adam(),
    optax.add_decayed_weights(weight_decay),
    optax.scale(-learning_rate),
)
params = variables['params']
opt_state = tx.init(params)

for batch in get_ds_train():
  params, opt_state = train_step(params, opt_state, batch)

eval(params)
```

+++ {"id": "MZE97FEbCgmn"}

### Gradient Clipping

```{code-cell}
:id: Y_DCoogODZL6

grad_clip_norm = 1.0
```

```{code-cell}
:id: mFnX8fb3Chwb
:outputId: aae283d2-623e-4a43-c001-3e62b11483ae

@jax.jit
def train_step(optimizer, batch):
  grads = jax.grad(loss)(optimizer.target, batch)
  grads_flat, _ = jax.tree_util.tree_flatten(grads)
  global_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads_flat]))
  g_factor = jnp.minimum(1.0, grad_clip_norm / global_l2)
  grads = jax.tree_util.tree_map(lambda g: g * g_factor, grads)
  return optimizer.apply_gradient(grads)

optimizer = flax.optim.Momentum(learning_rate, momentum).create(
    variables['params'])
for batch in get_ds_train():
  optimizer = train_step(optimizer, batch)

eval(optimizer.target)
```

```{code-cell}
:id: aJYN2A-TDhp3
:outputId: a9563d3a-7dc6-4ecf-bb7b-49356cf9fe13

@jax.jit
def train_step(params, opt_state, batch):
  grads = jax.grad(loss)(params, batch)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state

tx = optax.chain(
    optax.clip_by_global_norm(grad_clip_norm),
    optax.trace(decay=momentum),
    optax.scale(-learning_rate),
)
params = variables['params']
opt_state = tx.init(params)

for batch in get_ds_train():
  params, opt_state = train_step(params, opt_state, batch)

eval(params)
```

+++ {"id": "d9e9YmNHE-xV"}

### Learning Rate Schedules

```{code-cell}
:id: zCqz6fDsFB5n

schedule = lambda step: learning_rate * jnp.exp(step * 1e-3)
```

```{code-cell}
:id: NinuzivVFYb5
:outputId: c90b880e-d6f0-40fc-94b8-28d0622d8440

@jax.jit
def train_step(step, optimizer, batch):
  grads = jax.grad(loss)(optimizer.target, batch)
  return step + 1, optimizer.apply_gradient(grads, learning_rate=schedule(step))

optimizer = flax.optim.Momentum(learning_rate, momentum).create(
    variables['params'])
step = jnp.array(0)
for batch in get_ds_train():
  step, optimizer = train_step(step, optimizer, batch)

eval(optimizer.target)
```

```{code-cell}
:id: lzaYwXzuFp-L
:outputId: 0ed7f41f-cc2d-46c6-ae9f-4be5b906fb7f

@jax.jit
def train_step(params, opt_state, batch):
  grads = jax.grad(loss)(params, batch)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state

tx = optax.chain(
    optax.trace(decay=momentum),
    optax.scale_by_schedule(lambda step: -schedule(step)),
)
params = variables['params']
opt_state = tx.init(params)

for batch in get_ds_train():
  params, opt_state = train_step(params, opt_state, batch)

eval(params)
```

+++ {"id": "HLkQDKK0GCHH"}

### Multiple Optimizers

```{code-cell}
:id: d-2veL8lGDbV
:outputId: b006d3cc-eff6-410a-e201-ccd9464db9d7

@jax.jit
def train_step(optimizer, batch):
  grads = jax.grad(loss)(optimizer.target, batch)
  return optimizer.apply_gradient(grads)

kernels = flax.traverse_util.ModelParamTraversal(lambda p, _: 'kernel' in p)
biases = flax.traverse_util.ModelParamTraversal(lambda p, _: 'bias' in p)
kernel_opt = flax.optim.Momentum(learning_rate, momentum)
bias_opt = flax.optim.Momentum(learning_rate * 0.1, momentum)
optimizer = flax.optim.MultiOptimizer(
    (kernels, kernel_opt),
    (biases, bias_opt)
).create(variables['params'])

for batch in get_ds_train():
  optimizer = train_step(optimizer, batch)

eval(optimizer.target)
```

```{code-cell}
:id: MvQlkiCuHr41
:outputId: 88b4b0dd-4b80-4c5b-c21d-803c097b8cc6

@jax.jit
def train_step(params, opt_state, batch):
  grads = jax.grad(loss)(params, batch)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state

kernels = flax.traverse_util.ModelParamTraversal(lambda p, _: 'kernel' in p)
biases = flax.traverse_util.ModelParamTraversal(lambda p, _: 'bias' in p)

all_false = jax.tree_util.tree_map(lambda _: False, params)
kernels_mask = kernels.update(lambda _: True, all_false)
biases_mask = biases.update(lambda _: True, all_false)

tx = optax.chain(
    optax.trace(decay=momentum),
    optax.masked(optax.scale(-learning_rate), kernels_mask),
    optax.masked(optax.scale(-learning_rate * 0.1), biases_mask),
)
params = variables['params']
opt_state = tx.init(params)

for batch in get_ds_train():
  params, opt_state = train_step(params, opt_state, batch)

eval(params)
```

```{code-cell}
:id: v2omX108JYDo
:outputId: a9f45c8c-0db5-4b5e-b429-e6d79b22eca0

@jax.jit
def train_step(params, opt_state, batch):
  grads = jax.grad(loss)(params, batch)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state

kernels = flax.traverse_util.ModelParamTraversal(lambda p, _: 'kernel' in p)
biases = flax.traverse_util.ModelParamTraversal(lambda p, _: 'bias' in p)

all_false = jax.tree_util.tree_map(lambda _: False, params)
kernels_mask = kernels.update(lambda _: True, all_false)
biases_mask = biases.update(lambda _: True, all_false)

tx = optax.chain(
    optax.trace(decay=momentum),
    optax.multi_transform({
      'kernels': optax.scale(-learning_rate),
      'biases': optax.scale(-learning_rate * 0.1),
  }, kernels.update(lambda _: 'kernels',
                    biases.update(lambda _: 'biases', params))),
)
params = variables['params']
opt_state = tx.init(params)

for batch in get_ds_train():
  params, opt_state = train_step(params, opt_state, batch)

eval(params)
```
