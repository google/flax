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

  def __call__(self, x, y):
    return self.loss(self.features(x), self.features(y))

# Instantiate the model.
rngs = nnx.Rngs(0)
model = Foo(rngs=rngs)

# Dummy input for testing
x, y = rngs.normal((2,8))
```

Here, `self.sow` will store intermediate values under the key `'features'` in a collection associated with the
`nnx.Intermediate` type. If you want to log values to multiple different collections, you can use different subclasses of `nnx.Intermediate`
for each collection.

Now, we can wrap the module with the `nnx.capture` decorator, which wraps any `Callable` accepting a module as its argument (which includes `nnx.Module`s, their methods, or ordinary functions) to return both the resulting loss as well as any intermediate values stored to the `nnx.Intermediate` collection:

```python
capturing_model = nnx.capture(model, nnx.Intermediate)
result, intms = capturing_model(x, y)
jax.tree.map(lambda a: a.shape, intms)
```

Note that, by default, sow appends values every time it is called. We can see
this in the *features* intermediate values logged above. It contains a tuple with one element for the call on `x` and one for the call on `y`. To override the default append behavior, specify `init_fn` and `reduce_fn` - see `Module.sow()`.

## How `nnx.capture` Works

`nnx.capture` works by temporarily installing a set of mutable capture buffers on every module in the graph before calling the wrapped function, then harvesting those buffers afterward. Before calling the wrapped function, `capture` walks the entire module graph with `iter_modules`. For each module it sets a `__captures__` attribute: a tuple of Variable instances, one per requested `var_type`. Each Variable holds a plain `dict` that maps sow-key → accumulated value.

We can see this `__captures__` tuple by printing out the module contents during a `nnx.capture` call:

```python
@nnx.capture(nnx.Intermediate)
def print_captures(model):
      print("Captures:", model.__captures__)
_, intms = print_captures(nnx.Module())
```

`Module.sow` looks for the Variable in the `__captures__` tuple whose type matches `variable_type`, then writes its value into that dict using `reduce_fn`.

If no matching type is found, `sow` silently returns `False` without logging the value. This can be used to capture only a subset of the sown values. For example:

```python
class Metric1(nnx.Intermediate):
    pass

class Metric2(nnx.Intermediate):
    pass

@nnx.capture(Metric1)
def get_captures(model):
    model.sow(Metric1, 'gets_sown', jnp.ones(2))
    model.sow(Metric2, 'gets_ignored', jnp.ones(2))
_, intms = get_captures(nnx.Module())
jax.tree.map(lambda a: a.shape, intms)
```

## Capturing all intermediate values

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

  def __call__(self, x, y):
    return self.loss(self.features(x), self.features(y))

model = Foo(rngs=nnx.Rngs(0))
capturing_model = nnx.capture(model, nnx.Intermediate, method_outputs=nnx.Intermediate)
result, intms = capturing_model(x, y)
jax.tree.map(lambda a: a.shape, intms)
```

This pattern should be considered the "sledge hammer" approach to capturing intermediates. As a debugging and inspection tool it is very useful, but using the other patterns described in this guide will give you more fine-grained control over what intermediates you want to extract. We can also combine the `method_output_type` argument with manual calls to sow to capture both layer outputs and computations mid-layer.

## Extracting gradients of intermediate values

For debugging purposes, it can be useful to extract the gradients of intermediate values. This is a little tricky: jax doesn't have a stable mechanism for sowing information from the backward pass into to objects from the forward pass. Instead, we record the gradients of intermediate values using the `Module.perturb` method as follows:

```python
class Model(nnx.Module):
  def __call__(self, x):
    x2 = self.perturb('grad_of_x', x)
    self.sow(nnx.Intermediate, 'activations', x2)
    return 3 * x2

model = Model()

def train_step(model, x):
    _, perturbations = nnx.capture(model, nnx.Perturbation)(x)
    def loss(model, perturbations, x):
        return nnx.capture(model, nnx.Intermediate, init=perturbations)(x)

    (grads, perturb_grads), sowed = nnx.grad(loss, argnums=(0, 1), has_aux=True)(model, perturbations, x)
    return nnx.merge_state(perturb_grads, sowed)

train_step(model, 1.0)
```

There are four steps:

**Step One: Initialize *perturbations* of the model**.

We do this with a call to `nnx.capture(model, nnx.Perturbation)`. Before the call, `capture` installs `__captures__` on the module — a tuple containing one empty `Perturbation` buffer (as described in "How `nnx.capture` Works" above). When `self.perturb` runs, it checks `__captures__` for a matching `Perturbation` Variable, initialises the slot to `zeros_like(value)`, and returns `zeros + x`. After the call, `__captures__` is removed and the filled buffer is returned as `perturbations`.

```python
class Model(nnx.Module):
  def __call__(self, x):
    print("before perturb:", self.__captures__)
    x2 = self.perturb('grad_of_x', x)
    print("after  perturb:", self.__captures__)
    self.sow(nnx.Intermediate, 'activations', x2)
    # sow is a no-op: Intermediate is not in __captures__, so it returns False silently
    return 3 * x2

model = Model()
_, perturbations = nnx.capture(model, nnx.Perturbation)(1.0)
print(perturbations)
```

There are only two differences between `sow` and `perturb`:

- The `nnx.Variable` tag used for values written with `self.perturb` is `nnx.Perturbation` rather than `nnx.Intermediate`.
    
- `perturb` returns the logged value. You must use this returned value rather than the original value for the gradient capturing machinery to work.

The `var_types` argument to `capture` restricts which of the logged values we want to return. Because we only want the intermediates logged with `self.perturb` statements, we only capture `nnx.Perturbation` types.

**Step Two: Run the model again, but add in these perturbations**.

Call `capture` again with `init=perturbations`. `capture` first builds a mapping from module path to the Variables in `init`, then uses it to pre-populate `__captures__`. Now `__captures__` has *two* buffers: an empty `Intermediate` buffer (from `var_types`) and a `Perturbation` buffer pre-populated from `init`. `self.perturb` finds the pre-populated buffer and returns `x + perturbation`; `self.sow` writes into the `Intermediate` buffer as normal.

```python
class Model(nnx.Module):
  def __call__(self, x):
    print("before perturb:", self.__captures__)
    x2 = self.perturb('grad_of_x', x)
    self.sow(nnx.Intermediate, 'activations', x2)
    print("after  sow:    ", self.__captures__)
    return 3 * x2

model = Model()
_, interms = nnx.capture(model, nnx.Intermediate, init=perturbations)(1.0)
```

This changes the behavior of `x2 = self.perturb('name', x)` to essentially be `x2 = x + perturbations['name']`. The gradient of our output with respect to `x` will be the same as the gradient with respect to the perturbation, because JAX can differentiate through the addition with respect to the perturbation value stored in the capture dict.

**Step Three: Take gradients**.

Specifically, take the gradient of this second `capture` call with respect to the perturbation arguments. JAX traces through exactly the same `__captures__` setup as Step Two, but with abstract (traced) array values instead of concrete ones. This will give us the same values as the gradients with respect to the intermediate variables. If we want to track intermediate variables in the forward pass at the same time, we'll need to return the intermediate values output of the `capture` call as well, so we'll need to pass `has_aux=True` to `nnx.grad`.

**Step Four: Combine intermediate states**

Merge the `State` object we get from the perturbation gradients with the `State` object for forward intermediates with `nnx.merge_state(perturb_grads, sowed)`. At this point `__captures__` no longer exists on any module — it was cleaned up at the end of the `capture` call in Step Three.


## NNX Transforms and Capturing

`nnx.capture` composes with NNX transforms such as `nnx.vmap`. The main thing to keep in mind is that perturbations must be initialized with a run that has the same batch structure as the training step that will consume them.

Consider a model that calls both `sow` and `perturb`:

```python
class Foo(nnx.Module):
  def __init__(self, dim):
    self.w = nnx.Param(jax.random.normal(jax.random.key(0), dim))

  def __call__(self, x):
    x = self.perturb('grad_of_x', x)
    y = jnp.dot(x, self.w)
    self.sow(nnx.Intermediate, 'y', y)
    return y
```

The training step vmaps `loss_grad` over a batch of inputs and perturbations, while the model weights are shared across the batch (`in_axes=None`):

```python
@nnx.jit
def train_step(model, x):
  _, perturbations = init_perturbations(model, x)
  def loss_grad(model, perturbations, x):
    def loss(model, perturbations, x):
      loss, interms = nnx.capture(model, nnx.Intermediate, init=perturbations)(x)
      return loss, interms
    (grads, perturb_grads), sowed = nnx.grad(loss, argnums=(0, 1), has_aux=True)(model, perturbations, x)
    return grads, nnx.merge_state(perturb_grads, sowed)
  return nnx.vmap(loss_grad, in_axes=(None, 0, 0))(model, perturbations, x)
```

After every training step, we can sum the gradients and pass them to an `Optimizer` to adjust the model, as usual. But we can also look at the full batch of sown values and perturbations.

Because `train_step` expects `perturbations` to have a leading batch axis (axis 0), the perturbation initialization run must also produce a batched `perturbations` state. We do this inside an `init_perturbations` method that splits the model and vmaps the run with `in_axes=(0, None, 0)` for `(intermediates, params, x)`.

```python
@nnx.capture(nnx.Perturbation)
def init_perturbations(model, x):
    graphdef, intms, params = nnx.split(model, nnx.Intermediate, nnx.Param)
    def forward(intms, params, x):
      return nnx.merge(graphdef, intms, params)(x)
    return nnx.vmap(forward, in_axes=(0, None, 0))(intms, params, x)
```

Putting it together:

```python
model, x = Foo(4), jnp.ones((3, 4))
_, intermediates = train_step(model, x)
jax.tree.map(lambda a: a.shape, intermediates)
```

The pattern generalises: whenever a transform introduces a new batch axis over which `capture` runs, initialize perturbations with a matching vmapped pre-run so that the `init=perturbations` argument inside the transform has the correct shape.
