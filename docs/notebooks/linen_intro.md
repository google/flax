---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

+++ {"id": "C1QVJFlVsxcZ"}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/flax/blob/main/docs/notebooks/linen_intro.ipynb)
[![Open On GitHub](https://img.shields.io/badge/Open-on%20GitHub-blue?logo=GitHub)](https://github.com/google/flax/blob/main/docs/notebooks/linen_intro.ipynb)

# Preface

<br>
<div style="font-variant: small-caps;">CAVEAT PROGRAMMER</div>

The below is an alpha API preview and things might break.  The surface syntax of the features of the API are not fixed in stone, and we welcome feedback on any points.

+++ {"id": "23zkGDayszYI"}

## Useful links

⟶ [Slides](https://docs.google.com/presentation/d/1ngKWUwsSqAwPRvATG8sAxMzu9ujv4N__cKsUofdNno0/edit?usp=sharing) for the core ideas of the new Functional Core and Linen

⟶ "Design tests" guided our design process. Many are available for [functional core](https://github.com/google/flax/tree/main/examples/core_design_test) and some for the [proposed Module abstraction](https://github.com/google/flax/tree/main/examples/linen_design_test/)

⟶ Ported examples: [ImageNet](https://github.com/google/flax/tree/main/examples/imagenet) and [WMT](https://github.com/google/flax/tree/main/examples/wmt) (to the proposed Module abstraction). TODO: Port to functional core.

⟶ Our new [discussion forums](https://github.com/google/flax/discussions/)

+++ {"id": "vGtC_5W4mQnY"}

# Install and Import

```{code-cell}
:id: HgRZ_G8wGcoB
:tags: [skip-execution]

# Install the newest JAXlib version.
!pip install --upgrade -q pip jax jaxlib
# Install Flax at head:
!pip install --upgrade -q git+https://github.com/google/flax.git
```

```{code-cell}
:id: Kvx7GmavHZbD

import functools
from typing import Any, Callable, Sequence, Optional
import jax
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn
```

+++ {"id": "u86fYsrEfYow"}

# Invoking Modules

+++ {"id": "nrVbFrh1ffve"}

Let's instantiate a `Dense` layer.
 - Modules are actually objects in this API, so we provide _contructor arguments_ when initializing the Module.  In this case, we only have to provide the output `features` dimension.

```{code-cell}
:id: EcDH20Uufc-v

model = nn.Dense(features=3)
```

+++ {"id": "hL4NgtBwgI0S"}

We need to initialize the Module variables, these include the parameters of the Module as well as any other state variables.

We call the `init` method on the instantiated Module.  If the Module `__call__` method has args `(self, *args, **kwargs)` then we call `init` with `(rngs, *args, **kwargs)` so in this case, just `(rng, input)`:

```{code-cell}
:id: Vjx0HWNcfa8h
:outputId: 3adfaeaf-977e-4e82-8adf-d254fae6eb91

# Make RNG Keys and a fake input.
key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

# provide key and fake input to get initialized variables
init_variables = model.init(key2, x)

init_variables
```

+++ {"id": "ubFTzroGhErh"}

We call the `apply` method on the instantiated Module.  If the Module `__call__` method has args `(self, *args, **kwargs)` then we call `apply` with `(variables, *args, rngs=<RNGS>, mutable=<MUTABLEKINDS>, **kwargs)` where
 - `<RNGS>` are the optional _call time_ RNGs for things like dropout. For simple Modules this is just a single key, but if your module has multiple __kinds__ of data, it's a dictionary of rng-keys per-kind, e.g. `{'params': key0, 'dropout': key1}` for a Module with dropout layers.
 - `<MUTABLEKINDS>` is an optional list of names of __kinds__ that are expected to be mutated during the call. e.g. `['batch_stats']` for a layer updating batchnorm statistics.

So in this case, just `(variables, input)`:

```{code-cell}
:id: R9QZ6EOBg5X8
:outputId: e8c389a6-29f3-4f93-97ea-703e85a8b811

y = model.apply(init_variables, x)
y
```

+++ {"id": "lNH06qc1hPrd"}

Additional points:
 - If you want to `init` or `apply` a Module using a method other than call, you need to provide the `method=` kwarg to `init` and `apply` to use it instead of the default `__call__`, e.g. `method='encode'`, `method='decode'` to apply the encode/decode methods of an autoencoder.

+++ {"id": "jjsyiBjIYcAB"}

# Defining Basic Modules

+++ {"id": "UvU7416Ti_lR"}

## Composing submodules

+++ {"id": "LkTy0hmJdE5G"}

We support declaring modules in `setup()` that can still benefit from shape inference by using __Lazy Initialization__ that sets up variables the first time the Module is called.

```{code-cell}
:id: qB6l-9EabOwH
:outputId: 1a6c6a17-0b95-42c2-b5bf-b9ad80fd7758
:tags: []

class ExplicitMLP(nn.Module):
  features: Sequence[int]

  def setup(self):
    # we automatically know what to do with lists, dicts of submodules
    self.layers = [nn.Dense(feat) for feat in self.features]
    # for single submodules, we would just write:
    # self.layer1 = nn.Dense(feat1)

  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate(self.layers):
      x = lyr(x)
      if i != len(self.layers) - 1:
        x = nn.relu(x)
    return x

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = ExplicitMLP(features=[3,4,5])
init_variables = model.init(key2, x)
y = model.apply(init_variables, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(init_variables)))
print('output:\n', y)
```

+++ {"id": "slwE6ULqc_t_"}

Here we show the equivalent compact form of the MLP that declares the submodules inline using the `@compact` decorator.

```{code-cell}
:id: UPNGIr6wcGaw
:outputId: b3709789-e66e-4e20-f6b2-04022f8a62bb
:tags: []

class SimpleMLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
      # providing a name is optional though!
      # the default autonames would be "Dense_0", "Dense_1", ...
      # x = nn.Dense(feat)(x)
    return x

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = SimpleMLP(features=[3,4,5])
init_variables = model.init(key2, x)
y = model.apply(init_variables, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(init_variables)))
print('output:\n', y)
```

+++ {"id": "b2OzKXYyjFSf"}

## Declaring and using variables

+++ {"id": "uYwS5KbcmYIp"}

Flax uses lazy initialization, which allows declared variables to be initialized only at the first site of their use, using whatever shape information is available a the local call site for shape inference.  Once a variable has been initialized, a reference to the data is kept for use in subsequent calls.

For declaring parameters that aren't mutated inside the model, but rather by gradient descent, we use the syntax:

 `self.param(parameter_name, parameter_init_fn, *init_args)`

with arguments:
 - `parameter_name` just the name, a string
 - `parameter_init_fn` a function taking an RNG key and a variable number of other arguments, i.e. `fn(rng, *args)`. typically those in `nn.initializers` take an `rng` and a `shape` argument.
 - the remaining arguments to feed to the init function when initializing.

Again, we'll demonstrate declaring things inline as we typically do using the `@compact` decorator.

```{code-cell}
:id: 7OACbTFHjMvl
:outputId: bc5cb1f2-c5e9-4159-d131-73247009e32f
:tags: []

class SimpleDense(nn.Module):
  features: int
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs):
    kernel = self.param('kernel',
                        self.kernel_init,  # RNG passed implicitly.
                        (inputs.shape[-1], self.features))  # shape info.
    y = lax.dot_general(inputs, kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),)
    bias = self.param('bias', self.bias_init, (self.features,))
    y = y + bias
    return y

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = SimpleDense(features=3)
init_variables = model.init(key2, x)
y = model.apply(init_variables, x)

print('initialized parameters:\n', init_variables)
print('output:\n', y)
```

+++ {"id": "KgEwkrkfdlt8"}

We can also declare variables in setup, though in doing so you can't take advantage of shape inference and have to provide explicit shape information at initialization.  The syntax is a little repetitive in this case right now, but we do force agreement of the assigned names.

```{code-cell}
:id: CE0CTLVvZ8Yn
:outputId: 1e822bd8-7a08-4e80-e0e6-a86637c46772
:tags: []

class ExplicitDense(nn.Module):
  features_in: int  # <-- explicit input shape
  features: int
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros_init()

  def setup(self):
    self.kernel = self.param('kernel',
                             self.kernel_init,
                             (self.features_in, self.features))
    self.bias = self.param('bias', self.bias_init, (self.features,))

  def __call__(self, inputs):
    y = lax.dot_general(inputs, self.kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),)
    y = y + self.bias
    return y

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = ExplicitDense(features_in=4, features=3)
init_variables = model.init(key2, x)
y = model.apply(init_variables, x)

print('initialized parameters:\n', init_variables)
print('output:\n', y)
```

+++ {"id": "t4MVj1RBmxsZ"}

## General Variables

+++ {"id": "CJatarOTpByQ"}

For declaring generally mutable _variables_ that may be mutated inside the model we use the call:

 `self.variable(variable_kind, variable_name, variable_init_fn, *init_args)`

with arguments:
 - `variable_kind` the "kind" of state this variable is, i.e. the name of the nested-dict collection that this will be stored in inside the top Modules variables.  e.g. `batch_stats` for the moving statistics for a batch norm layer or `cache` for autoregressive cache data.  Note that parameters also have a kind, but they're set to the default `param` kind.
 - `variable_name` just the name, a string
 - `variable_init_fn` a function taking a variable number of other arguments, i.e. `fn(*args)`. Note that we __don't__ assume the need for an RNG, if you _do_ want an RNG, provide it via a `self.make_rng(variable_kind)` call in the provided arguments.
 - the remaining arguments to feed to the init function when initializing.

⚠️ Unlike parameters, we expect these to be mutated, so `self.variable` returns not a constant, but a _reference_ to the variable.  To __get__ the raw value, you'd write `myvariable.value` and to __set__ it `myvariable.value = new_value`.

```{code-cell}
:id: u6_fbrW2XT5t
:outputId: 2a8f5453-81b1-44dc-a431-d14b372c5710
:tags: []

class Counter(nn.Module):
  @nn.compact
  def __call__(self):
    # easy pattern to detect if we're initializing
    is_initialized = self.has_variable('counter', 'count')
    counter = self.variable('counter', 'count', lambda: jnp.zeros((), jnp.int32))
    if is_initialized:
      counter.value += 1
    return counter.value


key1 = random.PRNGKey(0)

model = Counter()
init_variables = model.init(key1)
print('initialized variables:\n', init_variables)

y, mutated_variables = model.apply(init_variables, mutable=['counter'])

print('mutated variables:\n', mutated_variables)
print('output:\n', y)
```

+++ {"id": "VLxwg2aMxUmy"}

## Another Mutability and RNGs Example

+++ {"id": "NOARPIowyeXS"}

Let's make an artificial, goofy example that mixes differentiable parameters, stochastic layers, and mutable variables:

```{code-cell}
:id: BBrbcEdCnQ4o
:outputId: 8f299a5c-74c8-476c-93fa-e5543901ec45
:tags: []

class Block(nn.Module):
  features: int
  training: bool
  @nn.compact
  def __call__(self, inputs):
    x = nn.Dense(self.features)(inputs)
    x = nn.Dropout(rate=0.5)(x, deterministic=not self.training)
    x = nn.BatchNorm(use_running_average=not self.training)(x)
    return x

key1, key2, key3, key4 = random.split(random.PRNGKey(0), 4)
x = random.uniform(key1, (3,4,4))

model = Block(features=3, training=True)

init_variables = model.init({'params': key2, 'dropout': key3}, x)
_, init_params = init_variables.pop('params')

# When calling `apply` with mutable kinds, returns a pair of output,
# mutated_variables.
y, mutated_variables = model.apply(
    init_variables, x, rngs={'dropout': key4}, mutable=['batch_stats'])

# Now we reassemble the full variables from the updates (in a real training
# loop, with the updated params from an optimizer).
updated_variables = freeze(dict(params=init_params,
                                **mutated_variables))

print('updated variables:\n', updated_variables)
print('initialized variable shapes:\n',
      jax.tree_util.tree_map(jnp.shape, init_variables))
print('output:\n', y)

# Let's run these model variables during "evaluation":
eval_model = Block(features=3, training=False)
y = eval_model.apply(updated_variables, x)  # Nothing mutable; single return value.
print('eval output:\n', y)
```

+++ {"id": "Lcp28h72810L"}

# JAX transformations inside modules

+++ {"id": "WEpbn8si0ATT"}

## JIT

+++ {"id": "-k-5gXTJ0EpD"}

It's not immediately clear what use this has, but you can compile specific submodules if there's a reason to.

_Known Gotcha_: at the moment, the decorator changes the RNG stream slightly, so comparing jitted an unjitted initializations will look different.

```{code-cell}
:id: UEUTO8bf0Kf2
:outputId: 3f324d0f-259f-40f0-8273-103f7fc281c5
:tags: []

class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      # JIT the Module (it's __call__ fn by default.)
      x = nn.jit(nn.Dense)(feat, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x

key1, key2 = random.split(random.PRNGKey(3), 2)
x = random.uniform(key1, (4,4))

model = MLP(features=[3,4,5])
init_variables = model.init(key2, x)
y = model.apply(init_variables, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(init_variables)))
print('output:\n', y)
```

+++ {"id": "D1tfTdRjyJYK"}

## Remat

+++ {"id": "goiHMi4qyLiZ"}

For memory-expensive computations, we can `remat` our method to recompute a Module's output during a backwards pass.

_Known Gotcha_: at the moment, the decorator changes the RNG stream slightly, so comparing remat'd and undecorated initializations will look different.

```{code-cell}
:id: sogMxDQpyMZE
:outputId: 7fe8e13b-7dd6-4e55-ee50-ce334e8ed178
:tags: []

class RematMLP(nn.Module):
  features: Sequence[int]
  # For all transforms, we can annotate a method, or wrap an existing
  # Module class. Here we annotate the method.
  @nn.remat
  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x

key1, key2 = random.split(random.PRNGKey(3), 2)
x = random.uniform(key1, (4,4))

model = RematMLP(features=[3,4,5])
init_variables = model.init(key2, x)
y = model.apply(init_variables, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(init_variables)))
print('output:\n', y)
```

+++ {"id": "l0pJtxVwyCgp"}

## Vmap

+++ {"id": "TqVbjhOkyEaj"}

You can now `vmap` Modules inside.  The transform has a lot of arguments, they have the usual jax vmap args:
 - `in_axes` - an integer or `None` for each input argument
 - `out_axes` - an integer or `None` for each output argument
 - `axis_size` - the axis size if you need to give it explicitly

In addition, we provide for each __kind__ of variable it's axis rules:

 - `variable_in_axes` - a dict from kinds to a single integer or `None` specifying the input axes to map
 - `variable_out_axes` - a dict from kinds to a single integer or `None` specifying the output axes to map
 - `split_rngs` - a dict from RNG-kinds to a bool, specifying whether to split the rng along the axis.


Below we show an example defining a batched, multiheaded attention module from a single-headed unbatched attention implementation.

```{code-cell}
:id: PIGiriD0yFXo
:outputId: 223d880e-c7b2-4210-ebb5-dbfcdd9aed09
:tags: []

class RawDotProductAttention(nn.Module):
  attn_dropout_rate: float = 0.1
  train: bool = False

  @nn.compact
  def __call__(self, query, key, value, bias=None, dtype=jnp.float32):
    assert key.ndim == query.ndim
    assert key.ndim == value.ndim

    n = query.ndim
    attn_weights = lax.dot_general(
        query, key,
        (((n-1,), (n - 1,)), ((), ())))
    if bias is not None:
      attn_weights += bias
    norm_dims = tuple(range(attn_weights.ndim // 2, attn_weights.ndim))
    attn_weights = jax.nn.softmax(attn_weights, axis=norm_dims)
    attn_weights = nn.Dropout(self.attn_dropout_rate)(attn_weights,
                                                      deterministic=not self.train)
    attn_weights = attn_weights.astype(dtype)

    contract_dims = (
        tuple(range(n - 1, attn_weights.ndim)),
        tuple(range(0, n  - 1)))
    y = lax.dot_general(
        attn_weights, value,
        (contract_dims, ((), ())))
    return y

class DotProductAttention(nn.Module):
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  train: bool = False

  @nn.compact
  def __call__(self, inputs_q, inputs_kv, bias=None, dtype=jnp.float32):
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    out_features = self.out_features or inputs_q.shape[-1]

    QKVDense = functools.partial(
      nn.Dense, features=qkv_features, use_bias=False, dtype=dtype)
    query = QKVDense(name='query')(inputs_q)
    key = QKVDense(name='key')(inputs_kv)
    value = QKVDense(name='value')(inputs_kv)

    y = RawDotProductAttention(train=self.train)(
        query, key, value, bias=bias, dtype=dtype)

    y = nn.Dense(features=out_features, dtype=dtype, name='out')(y)
    return y

class MultiHeadDotProductAttention(nn.Module):
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  batch_axes: Sequence[int] = (0,)
  num_heads: int = 1
  broadcast_dropout: bool = False
  train: bool = False
  @nn.compact
  def __call__(self, inputs_q, inputs_kv, bias=None, dtype=jnp.float32):
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    out_features = self.out_features or inputs_q.shape[-1]

    # Make multiheaded attention from single-headed dimension.
    Attn = nn.vmap(DotProductAttention,
                   in_axes=(None, None, None),
                   out_axes=2,
                   axis_size=self.num_heads,
                   variable_axes={'params': 0},
                   split_rngs={'params': True,
                               'dropout': not self.broadcast_dropout})

    # Vmap across batch dimensions.
    for axis in reversed(sorted(self.batch_axes)):
      Attn = nn.vmap(Attn,
                     in_axes=(axis, axis, axis),
                     out_axes=axis,
                     variable_axes={'params': None},
                     split_rngs={'params': False, 'dropout': False})

    # Run the vmap'd class on inputs.
    y = Attn(qkv_features=qkv_features // self.num_heads,
             out_features=out_features,
             train=self.train,
             name='attention')(inputs_q, inputs_kv, bias)

    return y.mean(axis=-2)


key1, key2, key3, key4 = random.split(random.PRNGKey(0), 4)
x = random.uniform(key1, (3, 13, 64))

model = functools.partial(
  MultiHeadDotProductAttention,
  broadcast_dropout=False,
  num_heads=2,
  batch_axes=(0,))

init_variables = model(train=False).init({'params': key2}, x, x)
print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(init_variables)))

y = model(train=True).apply(init_variables, x, x, rngs={'dropout': key4})
print('output:\n', y.shape)
```

+++ {"id": "U-bDSQElvM09"}

## Scan

+++ {"id": "8oiRXIC6xQ--"}

Scan allows us to apply `lax.scan` to Modules, including their parameters and mutable variables.  To use it we have to specify how we want each "kind" of variable to be transformed.  For scanned variables we specify similar to vmap via in `variable_in_axes`, `variable_out_axes`:
 - `nn.broadcast` broadcast the variable kind across the scan steps as a constant
 - `<axis:int>` scan along `axis` for e.g. unique parameters at each step

OR we specify that the variable kind is to be treated like a "carry" by passing to the `variable_carry` argument.

Further, for `scan`'d variable kinds, we further specify whether or not to split the rng at each step.

```{code-cell}
:id: oxA_lWm7tH2B
:outputId: 7d9ebed3-64de-4ca8-9dce-4b09ba9e31a1
:tags: []

class SimpleScan(nn.Module):
  @nn.compact
  def __call__(self, xs):
    dummy_rng = random.PRNGKey(0)
    init_carry = nn.LSTMCell.initialize_carry(dummy_rng,
                                              xs.shape[:1],
                                              xs.shape[-1])
    LSTM = nn.scan(nn.LSTMCell,
                   in_axes=1, out_axes=1,
                   variable_broadcast='params',
                   split_rngs={'params': False})
    return LSTM(name="lstm_cell")(init_carry, xs)

key1, key2 = random.split(random.PRNGKey(0), 2)
xs = random.uniform(key1, (1, 5, 2))

model = SimpleScan()
init_variables = model.init(key2, xs)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(init_variables)))

y = model.apply(init_variables, xs)
print('output:\n', y)
```
