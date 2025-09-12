---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
---

# Extracting intermediate values


This guide will show you how to extract intermediate values from a module.
Consider the simple CNN we used in the MNIST tutorial. 

```python
from flax import nnx
import jax
import jax.numpy as jnp
from functools import partial
```

```python
class CNN(nnx.Module):
  """A simple CNN model."""

  def __init__(self, *, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
    self.batch_norm1 = nnx.BatchNorm(32, rngs=rngs)
    self.dropout1 = nnx.Dropout(rate=0.025)
    self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
    self.batch_norm2 = nnx.BatchNorm(64, rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
    self.dropout2 = nnx.Dropout(rate=0.025)
    self.linear2 = nnx.Linear(256, 10, rngs=rngs)

  def __call__(self, x, rngs= None):
    x = self.avg_pool(nnx.relu(self.batch_norm1(self.dropout1(self.conv1(x), rngs=rngs))))
    x = self.avg_pool(nnx.relu(self.batch_norm2(self.conv2(x))))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = nnx.relu(self.dropout2(self.linear1(x), rngs=rngs))
    x = self.linear2(x)
    return x

# Instantiate the model.
model = CNN(rngs=nnx.Rngs(0))

# Dummy input for testing
y = jnp.ones((1, 28, 28, 1))
```

We don't have direct access to intermediate values of `x` in the `__call__` method. There are a few ways to expose them:


## Store intermediate values with `sow`

The CNN can be augmented with calls to ``sow`` to store intermediates.

```python
class CNN_Sow(CNN):
    def __call__(self, x, rngs= None):
        x = self.avg_pool(nnx.relu(self.batch_norm1(self.dropout1(self.conv1(x), rngs=rngs))))
        x = self.avg_pool(nnx.relu(self.batch_norm2(self.conv2(x))))
        x = x.reshape(x.shape[0], -1)  # flatten
        self.sow(nnx.Capture, 'features', x)
        x = nnx.relu(self.dropout2(self.linear1(x), rngs=rngs))
        x = self.linear2(x)
        return x

sow_model = CNN_Sow(rngs=nnx.Rngs(0))
```

To extract these intermediate values, we must call the module with the `nnx.capture_intermediates` function:

```python
result, intms = nnx.capture_intermediates(sow_model, y, nnx.Rngs(1))
intms
```

Note that, by default sow appends values every time it is called:
- This is necessary because once instantiated, a module could be called multiple times in its parent module, and we want to catch all the sowed values.
- Therefore you want to make sure that you do not feed intermediate values back into variables. Otherwise every call will increase the length of that tuple and trigger a recompile.
- To override the default append behavior, specify init_fn and reduce_fn - see `Module.sow()`.


## Refactor `__call__` into sub-methods

This is a useful pattern for cases where it's clear in what particular way you want to split model evaluation

```python
class CNN_Refactor(CNN):
    
    def features(self, x, rngs= None):
        x = self.avg_pool(nnx.relu(self.batch_norm1(self.dropout1(self.conv1(x), rngs=rngs))))
        x = self.avg_pool(nnx.relu(self.batch_norm2(self.conv2(x))))
        return x.reshape(x.shape[0], -1)
        
    def classifier(self, features, rngs=None):
        x = nnx.relu(self.dropout2(self.linear1(features), rngs=rngs))
        return self.linear2(x)

    def __call__(self, x, rngs=None):
        return self.classifier(self.features(x, rngs), rngs)

refactored_model = CNN_Refactor(rngs=nnx.Rngs(0))
refactored_model.features(y, nnx.Rngs(1))
```

To observe the output at each step of a refactored module, we can use `nnx.capture_intermediates` as before, but adding the `method_outputs=True` argument. This will automatically `sow` the output of each method, including methods of sub-modules. 

```python
result, intms = nnx.capture_intermediates(refactored_model, y, nnx.Rngs(1), method_outputs=True)
jax.tree.map(lambda a: a.shape, intms)
```

This pattern should be considered the “sledge hammer” approach to capturing intermediates. As a debugging and inspection tool it is very useful, but using the other patterns described in this guide will give you more fine-grained control over what intermediates you want to extract.


We can also combine the `method_outputs` flag with manual calls to sow to capture both layer outputs and computations mid-layer. 

```python
result, intms = nnx.capture_intermediates(sow_model, y, nnx.Rngs(1), method_outputs=True)
jax.tree.map(lambda a: a.shape, intms)
```

# Extracting gradients of intermediate values


For debugging purposes, it can be useful to extract the gradients of intermediate values. This is a little tricky: the `sow` method currently only works in forward passes, so we can't just define a custom vjp rule that sows the gradient. If we try to sow from within a vjp backward rule, the intermediate value we'll get back will in general be undefined. 

Instead, we record the gradients of intermediate values using the `Module.perturb` method as follows:

```python
class Model(nnx.Module):
  def __call__(self, x):
    x2 = self.perturb('grad_of_x', x)
    return 3 * x2
    self.sow(nnx.Intermediate, 'y', y)
    return y

model = Model()

def train_step(model, perturbations, x):
  def loss(model, perturbations, x):
    return nnx.capture_intermediates(model, x, state=perturbations, filter=nnx.Not(nnx.Perturbation))

  (grads, perturbations), sowed = nnx.grad(loss, argnums=(0, 1), has_aux=True)(model, perturbations, x)
  return nnx.merge_state(sowed, perturbations)

x = 1.0
_, perturbations = nnx.capture_intermediates(model, x, filter=nnx.Perturbation)
metrics = train_step(model, perturbations, x)
```

<!-- #region -->
There are four steps:

**Step One: Initialize *perturbations* of the model**.

We do this with a call to `nnx.capture_intermediates(model, x, filter=nnx.Perturbation)`. Here, calls to `self.perturb` within the model are exactly the same as calls to `self.sow`: we are recording the value of the variables in the *forward* direction. There are only two differences:
    - The `nnx.Variable` tag used for values written with `self.sow` is an `nnx.Perturbation` rather than an `nnx.Intermediate`.
    - `perturb` returns the logged value. You must use this returned value rather than the original value for the gradient capturing machinery to work.

The `filter` argument to `capture_intermediates` restricts which of the logged values we want to return. Because we only want the intermediates logged with `self.perturb` statements, we filter out anything that's not a `nnx.Perturbation`. 

**Step Two: Run the model again, but add in these perturbations**.

Specifically, call `capture_intermediates` again with the argument `state=perturbations`. This changes the behavior of `x2 = self.perturb('name', x)` to essentially be `x2 = x + perturbations['name']`. The gradient of our output with respect to `x` will be the same as the gradient with respect to the perturbation.

**Step Three: Take gradients**.

Specifically, take the gradient of this second `capture_intermediates` call with respect to the perturbation arguments. This will give us the same values as the gradients with respect to the intermediate variables. If we want to track intermediate variables in the forward pass at the same time, we'll need to return the `sowed` output of the `capture_intermediates` call as well, so we'll need to pass `has_aux=True` to `nnx.grad`.


**Step Four: Combine intermediate states**

Merge the `State` object we get from the perturbation gradients with the `State` object for forward intermediates with `nnx.merge_state(sowed, perturbations)`. 
<!-- #endregion -->
