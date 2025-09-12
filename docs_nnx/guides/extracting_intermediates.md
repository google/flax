---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
---

# Extracting intermediate values

This guide will show you how to extract intermediate values from a module.
Consider a toy neural network with two pieces: a "feature" component that embeds
inputs in some feature space, and a "loss" component that operates on those features.
We'll want to log these feature components during training to identify any issues with
the feature extraction. To do this, we can use the `Module.sow` method.

```python
from flax import nnx
import jax
import jax.numpy as jnp
from functools import partial
```

```python
class Foo(nnx.Module):
  def __init__(self, *, rngs: nnx.Rngs):
    self.dense1 = nnx.Linear(8, 32, rngs=rngs)
    self.dense2 = nnx.Linear(32, 1, rngs=rngs)
    
  def features(self, x, rngs= None):
      feature = nnx.relu(self.dense1(x))
      self.sow(nnx.Intermediate, 'features', feature)
      return feature
        
  def loss(self, x_features, y_features):
    return jnp.sum((x_features - y_features)**2)

  @nnx.capture(nnx.Intermediate)
  def __call__(self, x, y):
    return self.loss(self.features(x), self.features(y))

# Instantiate the model.
model = Foo(rngs=nnx.Rngs(0))

# Dummy input for testing
x, y = nnx.Rngs.normal(nnx.Rngs(0), (2,8))
```

Here, `self.sow` will store intermediate values under the key `'features'` in a collection associated with the
`nnx.Intermediate` type. If you want to log values to multiple different collections, you can use different subclasses of `nnx.Intermediate`
for each collection.

The other important component to note in the example above is the use of the `nnx.capture(nnx.Intermediate)` decorator . This makes the `__call__` method return both the resulting loss as well as any intermediate values stored to the `nnx.Intermediate` collection:

```python
result, intms = model(x, y)
jax.tree.map(lambda a: a.shape, intms)
```

Note that, by default, sow appends values every time it is called. We can see
this in the *features* intermediate values logged above. It contains a tuple with one element for the call on `x` and one for the call on `y`. To override the default append behavior, specify `init_fn` and `reduce_fn` - see `Module.sow()`.

## Automatically capturing intermediate values

To observe the output of each method without manually adding calls to `sow`, we can call `nnx.capture` with the `method_outputs` argument. This will automatically `sow` the output of each method using the given variable type, including methods of sub-modules. 

```python
class Foo(nnx.Module):
  def __init__(self, *, rngs: nnx.Rngs):
    self.dense1 = nnx.Linear(8, 32, rngs=rngs)
    self.dense2 = nnx.Linear(32, 1, rngs=rngs)
    
  def features(self, x, rngs= None):
      feature = nnx.relu(self.dense1(x))
      return feature
        
  def loss(self, x_features, y_features):
    return jnp.sum((x_features - y_features)**2)

  @nnx.capture(nnx.Intermediate, method_outputs=nnx.Intermediate)
  def __call__(self, x, y):
    return self.loss(self.features(x), self.features(y))

model = Foo(rngs=nnx.Rngs(0))
result, intms = model(x, y)
jax.tree.map(lambda a: a.shape, intms)
```

This pattern should be considered the “sledge hammer” approach to capturing intermediates. As a debugging and inspection tool it is very useful, but using the other patterns described in this guide will give you more fine-grained control over what intermediates you want to extract. We can also combine the `method_output_type` argument with manual calls to sow to capture both layer outputs and computations mid-layer. 

## Extracting gradients of intermediate values

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
    return nnx.capture(model, nnx.Intermediate, init=perturbations)(x)

  (grads, perturb_grads), sowed = nnx.grad(loss, argnums=(0, 1), has_aux=True)(model, perturbations, x)
  return nnx.merge_state(perturb_grads, sowed)

x = 1.0
forward = nnx.capture(model, nnx.Perturbation)
_, perturbations = forward(x)
metrics = train_step(model, perturbations, x)
metrics
```

There are four steps:

**Step One: Initialize *perturbations* of the model**.

We do this with a call to `nnx.capture(model, nnx.Perturbation)`. Here, calls to `self.perturb` within the model are exactly the same as calls to `self.sow`: we are recording the value of the variables in the *forward* direction. There are only two differences:
    - The `nnx.Variable` tag used for values written with `self.sow` is an `nnx.Perturbation` rather than an `nnx.Intermediate`.
    - `perturb` returns the logged value. You must use this returned value rather than the original value for the gradient capturing machinery to work.

The `var_types` argument to `capture` restricts which of the logged values we want to return. Because we only want the intermediates logged with `self.perturb` statements, we only capture `nnx.Perturbation` types. 

**Step Two: Run the model again, but add in these perturbations**.

Specifically, call `capture` again with the argument `init=perturbations`. This changes the behavior of `x2 = self.perturb('name', x)` to essentially be `x2 = x + perturbations['name']`. The gradient of our output with respect to `x` will be the same as the gradient with respect to the perturbation.

**Step Three: Take gradients**.

Specifically, take the gradient of this second `capture` call with respect to the perturbation arguments. This will give us the same values as the gradients with respect to the intermediate variables. If we want to track intermediate variables in the forward pass at the same time, we'll need to return the intermediate values output of the `capture` call as well, so we'll need to pass `has_aux=True` to `nnx.grad`.


**Step Four: Combine intermediate states**

Merge the `State` object we get from the perturbation gradients with the `State` object for forward intermediates with `nnx.merge_state(sowed, perturbations)`.
