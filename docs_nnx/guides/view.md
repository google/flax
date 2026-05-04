---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Model Views
This guide covers how to use NNX Views, which are useful for handling state in layers like `Dropout` and `BatchNorm` which behave differently in training and evaluation. Similar to `.view` for numpy arrays, NNX Views allow you to modify static attributes of the model while still sharing the same data. For a quick intro, consider the following example showcasing `nnx.view`, the canonical NNX View that sets module modes.

For other views, NNX follows a naming convention for view-creating functions: names starting with `with_` return a new version of the input with modified module or variable attributes, while names starting with `as_` return a new tree with variables transformed into a different representation. In both cases the underlying JAX array data is shared with the original.

```{code-cell}
from flax import nnx

# example model with different train/eval behavior
rngs = nnx.Rngs(0)
model = nnx.Sequential(
  nnx.Linear(2, 4, rngs=rngs), nnx.BatchNorm(4, rngs=rngs), nnx.Dropout(0.1)
)

# set train and eval modes
train_model = nnx.view(model, deterministic=False, use_running_average=False)
eval_model = nnx.view(model, deterministic=True, use_running_average=True)

# Can see deterministic is different between train_model and eval_model
assert train_model.layers[2].deterministic == False
assert eval_model.layers[2].deterministic == True

# Weights are shared between the models
assert train_model.layers[0].kernel is eval_model.layers[0].kernel

# Print information about kwargs for nnx.view with nnx.view_info
print(nnx.view_info(model))
```

## Motivation

Some layers in ML inherently involve state. Consider for example the `nnx.Dropout` layer, which behaves differently during training and evaluation. In these different scenarios, we need a simple way to ensure that the model behaves as intended to avoid silent bugs. A common pattern in other frameworks is to mutate a single `model` object to switch between training and evaluation modes. This requires the programmer to remember to toggle modes in many places throughout the code, which can hurt readability and lead to subtle bugs when a mode switch is forgotten.

`nnx.view` offers a cleaner alternative: you declare the different model configurations once at the beginning of your code and then simply use the appropriate view wherever needed. Each view shares the same underlying weights, so parameter updates are automatically reflected across all views. We demonstrate this with a simple example below.

```{code-cell}
import jax
import optax
import matplotlib.pyplot as plt

in_dim, hidden_dim, out_dim = 16, 32, 2


class MyModel(nnx.Module):
  def __init__(
    self,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout_rate: float,
    *,
    rngs: nnx.Rngs,
  ):
    self.lin1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
    self.do = nnx.Dropout(dropout_rate)
    self.bn = nnx.BatchNorm(hidden_dim, rngs=rngs)
    self.lin2 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

  def __call__(self, x, *, rngs=None):
    x = nnx.relu(self.do(self.bn(self.lin1(x)), rngs=rngs))
    return self.lin2(x)
```

Lets take a look at the model to see what is going on.

```{code-cell}
# can display to inspect state
model = MyModel(in_dim, hidden_dim, out_dim, 0.1, rngs=nnx.Rngs(0))
nnx.display(model)

# can assert to inspect state
assert model.do.deterministic == False
```

From the model display, we can see that `Dropout` has `deterministic == False`, suggesting that the model is in training mode. In order to know this, we had to display the model and/or know that `Dropout` is set to training mode by default. It is not clear what state the model is in just by looking at the code without additional inspection. We instead want to be very explicit about what state the model is in. 

This is where `nnx.view` comes in. This function updates the modes for each submodule of a neural network based on the kwargs passed into the function. The underlying model weights are then shared between different views. We set up a training and evaluation version of the model below.

```{code-cell}
train_model = nnx.view(model, deterministic=False)
eval_model = nnx.view(model, deterministic=True)

# weights are references to the same data
assert train_model.lin1.kernel is eval_model.lin1.kernel

# Dropout.deterministic is different in each model
assert train_model.do.deterministic is False
assert eval_model.do.deterministic is True
```

## Example with `nnx.view`

+++

We first set up data generators and define train/eval step functions. The `train_step` receives an `nnx.Rngs` object for dropout randomness, while `eval_step` doesn't since dropout is disabled in `eval_model`.

```{code-cell}
ndata, batch_size, total_epochs, lr = 2048, 32, 100, 1e-3
rngs = nnx.Rngs(0)
x = rngs.normal((ndata, in_dim))
y = rngs.normal((ndata, out_dim))


@nnx.jit
def train_step(model, optimizer, x, y, rngs):
  def loss_fn(model, rngs):
    return ((model(x, rngs=rngs) - y) ** 2).mean()

  grads = nnx.grad(loss_fn)(model, rngs)
  optimizer.update(model, grads)


@nnx.jit
def eval_step(model, x, y):
  return ((model(x) - y) ** 2).mean()
```

Now we create `train_model` and `eval_model` views up front. During the training loop we simply use the appropriate view — there is no need to call `.train()` or `.eval()`, and it is always clear from the code which mode the model is in.

```{code-cell}
model = MyModel(in_dim, hidden_dim, out_dim, 0.1, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
train_model = nnx.view(model, deterministic=False)  # training view
eval_model = nnx.view(model, deterministic=True)  # eval view

eval_results = []
for epoch in range(total_epochs):
  for i in range(ndata // batch_size):
    idx = slice(i * batch_size, (i + 1) * batch_size)
    train_step(train_model, optimizer, x[idx], y[idx], rngs)  # use train_model

  eval_results.append(eval_step(eval_model, x, y))  # use eval_model
plt.plot(eval_results)
plt.show()
```

## Getting information with `nnx.view_info`
To see more information about the options for `nnx.view`, we can use the `nnx.view_info` function to display information about the arguments. This will display each submodule which contains a `set_view` method. It also provides information about the keyword arguments accepted by each submodule, including type information, default values, and docstring descriptions.

```{code-cell}
print(nnx.view_info(model))
```

## Writing modules compatible with `nnx.view`

You can make any custom module work with `nnx.view` by defining a `set_view` method. When `nnx.view` is called, it traverses the module tree and calls `set_view` on every submodule that defines one. `nnx.view` inspects the signature of each `set_view` method and only passes the keyword arguments that match the method's declared parameters. This means each module only receives the kwargs it cares about.

Your `set_view` method should follow these conventions:

1. **Accept keyword arguments with `None` defaults.** Each kwarg represents a configurable mode for this module. A `None` default means "leave unchanged", so views only override the modes you explicitly set.
2. **Only update the attribute when the kwarg is not `None`.** This ensures that unrelated views don't accidentally reset each other's settings.
3. **Include a Google-style docstring.** The `nnx.view_info` function parses these docstrings to display human-readable information about available view options.

The general pattern looks like this:

```python
class MyLayer(nnx.Module):
    ...

    def set_view(self, kwarg1: type1 = None, ..., kwargN: typeN = None):
        """Description of the module's configurable modes.

        Args:
          kwarg1: description of kwarg1.
          ...
          kwargN: description of kwargN.
        """
        if kwarg1 is not None:
            self.kwarg1 = kwarg1
        ...
```

Here is a concrete example — a `PrintLayer` that can be toggled to print a message during its forward pass:

```{code-cell}
class PrintLayer(nnx.Module):
  def __init__(self, msg: str | None = None):
    self.msg = msg

  def __call__(self, *args, **kwargs):
    if self.msg:
      print(self.msg)

  def set_view(self, msg: bool | None = None):
    """Example set_view docstring. This follows Google style docstrings.

    Args:
      msg: bool indicating if a message should be printed.
        If True, the `__call__` method prints the message.
    """
    if msg is not None:
      self.msg = msg


model = PrintLayer()
model_print = nnx.view(model, msg='Hello, World!')

model() # nothing printed
model_print() # prints "Hello, World!"
```

We can use `nnx.view_info` to inspect what view options `PrintLayer` exposes. This is especially handy when working with unfamiliar models — it lists every submodule that defines `set_view`, along with the accepted kwargs, their types, defaults, and docstring descriptions.

```{code-cell}
# Display the information for nnx.view
print(nnx.view_info(model))
```

The output shows that `PrintLayer` accepts a `msg` kwarg of type `bool` in its `set_view` method. When building larger models composed of many custom submodules, `nnx.view_info` gives you a quick summary of all the configurable modes across the entire module tree.

## Using `with_vars`

{func}`nnx.with_vars <flax.nnx.with_vars>` creates a view of a module tree by replacing ``Variable`` objects with copies that have different low-level JAX flags, while leaving the underlying array data shared. Unlike `view` and `with_attributes`, which change Python-level attributes on module objects, `with_vars` controls how ``Variable`` values are represented inside JAX.

The flags it controls are:

- **`ref`** — when `True`, each Variable's value is backed by a `jax.Ref`. This makes the module a valid pytree leaf for `jax.tree.map` and other JAX utilities that treat refs as mutable state.
- **`hijax`** — when `True`, Variables participate in JAX's *hijax* protocol and become first-class JAX values that can flow through `jax.grad`, `jax.jit`, and similar transforms without an explicit split/merge step.
- **`mutable`** — when `True`, marks Variables as mutable within a JAX transform.

The `only` argument accepts a {doc}`Filter <filters_guide>` to restrict which Variables are affected; unmatched Variables are returned as-is (shared with the original).

```{code-cell}
from flax import nnx
import jax
import jax.numpy as jnp

class SimpleModel(nnx.Module):
  def __init__(self, rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)

model = SimpleModel(nnx.Rngs(0))

# ref=True: expose Variable values as JAX refs so jax.tree.map can update them
ref_model = nnx.with_vars(model, ref=True)
ref_model = jax.tree.map(lambda x: x * 2, ref_model)

# The original model's kernel is unchanged; ref_model has doubled values
assert model.linear.kernel is not ref_model.linear.kernel
```

Use the `only` filter to convert only a subset of Variables:

```{code-cell}
# only convert Param variables, leave BatchStat variables unchanged
ref_params = nnx.with_vars(model, ref=True, only=nnx.Param)
```

## Using `with_attributes`

If you are working with modules that don't implement the `set_view` API, you can use {func}`nnx.with_attributes <flax.nnx.with_attributes>` to create views by directly replacing their attributes. Like `nnx.view`, it returns a new instance that shares jax arrays with the original, leaving the original unchanged.

```{code-cell}
class NoisyLinear(nnx.Module):
  def __init__(self, din, dout, *, training=None, rngs: nnx.Rngs):
    self.linear = nnx.Linear(din, dout, rngs=rngs)
    self.training = training

  def __call__(self, x, rngs=None):
    assert self.training is not None
    x = self.linear(x)
    if self.training:
      x = x + rngs.normal(x.shape) * 0.1
    return x

rngs = nnx.Rngs(0)
model = nnx.Sequential(
    NoisyLinear(4, 8, rngs=rngs),
    NoisyLinear(8, 2, rngs=rngs),
)

train_model = nnx.with_attributes(model, training=True)
eval_model = nnx.with_attributes(model, training=False)

print(f'{train_model.layer1.training=}')
y1 = train_model(jnp.ones((1, 4)), rngs=rngs)

print(f'{eval_model.layer1.training=}')
y2 = eval_model(jnp.ones((1, 4)))
```

## Using `recursive_map`

For more advanced transformations — such as replacing submodules — you can use {func}`nnx.recursive_map <flax.nnx.recursive_map>`. This function traverses the entire module tree bottom-up, calling a user-defined function `f(path, node)` on every node and leaf. Whatever `f` returns is used as the replacement for that node in the new tree. The resulting model view shares the Variables with the original (unless instructed otherwise).

In the example below, we use `recursive_map` to replace every `nnx.Linear` layer with a `NoisyLinear` version (reusing the class defined earlier) that adds random noise during training:

```{code-cell}
import jax.numpy as jnp

def add_noise(path, node):
  if isinstance(node, nnx.Linear):
    noisy = nnx.eval_shape(
      lambda: NoisyLinear(node.in_features, node.out_features, rngs=nnx.Rngs(0))
    )
    noisy.linear = node
    return noisy
  return node

rngs = nnx.Rngs(0)
model = nnx.Sequential(
    nnx.Linear(4, 8, rngs=rngs),
    nnx.Linear(8, 2, rngs=rngs),
)

noisy_model = nnx.recursive_map(add_noise, model)

y = noisy_model(jnp.ones((1, 4)), rngs=rngs)
print(noisy_model)s
```

Here `recursive_map` visited each node, and when it found an `nnx.Linear` instance it created a `NoisyLinear`, swapped in the original `Linear` as its inner layer, and returned it. The original `model` is unchanged and its weights are shared with `noisy_model`.

## Other NNX views

Several other NNX functions follow the `with_` / `as_` naming convention and produce views or transformed trees:

- {func}`nnx.as_pure <flax.nnx.as_pure>` — strips all ``Variable`` wrappers from a pytree and returns the raw inner values. This is useful for serialization or export, where Variable metadata is not needed.

  ```python
  _, state = nnx.split(model)
  pure_state = nnx.as_pure(state)  # Variable wrappers removed; plain arrays remain
  ```

- {func}`nnx.as_abstract <flax.nnx.as_abstract>` — annotates the abstract ``Variable`` objects produced by {func}`nnx.eval_shape` with sharding information derived from each Variable's `out_sharding` metadata. Used when working with JAX auto-sharding meshes.

  ```python
  with jax.set_mesh(mesh):
      abs_model = nnx.eval_shape(lambda: nnx.Linear(4, 8, rngs=nnx.Rngs(0)))
      abs_model = nnx.as_abstract(abs_model)  # sharding attached to abstract vars
  ```

- {func}`nnx.with_rngs <flax.nnx.rnglib.with_rngs>` — returns a copy of a pytree with ``RngStream`` objects split or forked according to filter rules. Used to prepare RNG state before JAX transforms like `vmap` that require per-device or per-replica keys.

  ```python
  # Split params stream into 4 keys (one per vmap replica); fork the rest
  vmapped_rngs = nnx.with_rngs(rngs, split={'params': 4}, fork=...)
  ```
