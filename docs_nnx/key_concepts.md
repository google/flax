# 🪄 Flax - Traps and Tricks 🪤

Flax is a **neural network library** built on top of JAX, a language for **accelerated numerical computations**. In effect, Flax is a pretty thin layer, and you likely will use some JAX APIs directly to do anything more than using the built-in Flax modules.

This means a **basic understanding on JAX helps you to use Flax well**. You would have better a mental model to understand what's happening underneath and how to debug a confusing error. This doc aims to clarify a few key concepts and help you build that uniquely-JAX mental model as a practical model developer (pun intended).

[JAX documentations](https://docs.jax.dev/en/latest/index.html) are great sources to learn more. We recommend all Flax users to at least read the [Key Concepts](https://docs.jax.dev/en/latest/key-concepts.html) doc.


```python
# For simulating multi-device environment
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
```

## What is JAX?

JAX is the lower level library that does **all the large-scale data computations**. It provides the singular data container, aka the `jax.Array`, and all the ways we possibly deal with them:

* **Make arithmetic operations upon the arrays**, including: the `jax.numpy` ops, automatic differentiation (`jax.grad`), batching (`jax.vmap`), and more.

* **Run computation on accelerators**, including: interface with various accelerator platforms and layouts; allocating buffers for arrays; compile and execute computation programs across accelerators.

* **Bundle multiple arrays together** using a simple concept called [pytrees](#pytrees).

This implies that any error related with accelerators and numericals are probably a JAX issue, or an issue with Flax built-in layers.

It also means you *can* build a neural network model with JAX alone, especially if you are comfortable with functional programming. JAX docsite have some [simple examples](https://docs.jax.dev/en/latest/notebooks/neural_network_with_tfds_data.html). The article [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/) also shows how to implement all the key elements of a GPT using JAX.


```python
def jax_linear(x, kernel, bias):
  return jnp.dot(x, kernel) + bias

params = {'kernel': jax.random.normal(jax.random.key(42), (4, 2)), 
          'bias': jnp.zeros((2,))}
x = jax.random.normal(jax.random.key(0), (2, 4))
y = jax_linear(x, params['kernel'], params['bias'])
```

## What is Flax?

Flax is a **neural network toolkit**, offering higher level abstractions that are handy for model developers. Such as:

* **Object-oriented `Module` class** to represent layers/models and bookkeep parameters.

* **Modeling utilities** like random number handling, model traversal and surgery, optimizers, advanced parameter bookkeeping, sharding annotations, and more.

* **Some built-in commonly-used** layers, initializers, and model examples.

Take the example below: A Flax layer `Linear`, during initialization, takes one RNG key and automatically initialize all internal parameters as `jax.Array`s. In forward pass, it carries out the exact same computation via JAX APIs.


```python
# Eligible parameters were created inside `linear`, using one RNG key 42
linear = nnx.Linear(in_features=4, out_features=2, rngs=nnx.Rngs(42))

# Flax created a `Param` wrapper over the actual `jax.Array` parameter to track metadata
print(type(linear.kernel))        # flax.nnx.Param
print(type(linear.kernel.value))  # jax.Array

# The computation of the two are the same
x = jax.random.normal(jax.random.key(0), (2, 4))
flax_y = linear(x)
jax_y = jax_linear(x, linear.kernel.value, linear.bias.value)
assert jnp.array_equal(flax_y, jax_y)
```

    <class 'flax.nnx.variablelib.Param'>
    <class 'jaxlib._jax.ArrayImpl'>


## Pytrees

Your code likely needs more than one `jax.Array`. A **pytree** is a container structure of multiple pytrees, possibly nested. It is a key and handly concept in the JAX world.

Many things are pytrees: Python dicts, lists, tuples, dataclasses, and more. The key is that a pytree can be "flattened" into multiple children, which are either pytrees or individual `jax.Array`s. JAX allows you to  register your own classes as pytree nodes.

Pytree is the primary data holder in JAX. When JAX transforms see a pytree argument, they automatically trace its internal `jax.Array`s when compiling. Therefore, it's crucial to organize your data as pytrees. [JAX pytree documentation](https://docs.jax.dev/en/latest/working-with-pytrees.html) has a thorough overview on pytrees and JAX APIs to manipulate them.

In Flax, a `Module` is not a pytree, but its state is a pytree. This how we implement Flax transforms (e.g. `flax.nnx.vmap` instead of `jax.vmap`) - the model is first converted to a JAX-compatible pytree dict, then the underlying JAX transforms are called.

In the future, Flax Modules will also become pytrees, allowing us to directly using JAX transforms.


```python
state = nnx.state(linear)
arrays, _ = jax.tree.flatten_with_path(state)
assert len(arrays) > 1
for kp, x in arrays:
  print(f'{jax.tree_util.keystr(kp)}: {x}')
```

    ['bias'].value: [0. 0.]
    ['kernel'].value: [[ 0.04119061 -0.2629074 ]
     [ 0.6772455   0.2807398 ]
     [ 0.16276604  0.16813846]
     [ 0.310975   -0.43336964]]



```python
# `nnx.Module` is not a pytree (yet). Therefore, flattening it will only return itself.
arrays, _ = jax.tree.flatten(linear)
assert len(arrays) == 1
assert arrays[0] == linear
```

## Traced vs. static data

A pytree *contains* JAX arrays, but a pytree is *more than* its JAX arrays. For example, a dictionary keeps information like the key of every array, and it might contain entries that are not JAX arrays. From JAX's standpoint, all data are one of the two types:

* **Traced** ("dynamic") data: JAX will trace them during compilation and optimize the operations upon them. If they stay inside a pytree argument, `jax.tree.flatten` must return them as leaves. They must be data values (`jax.Array`, Numpy array, scalar, etc).

* **"Static"** data: They stay as simple Python objects that don't get traced by JAX.

In practice, you would want to control what data goes into dynamic, and what to static. Dynamic data will be optimized by JAX, but you cannot base your code control flow upon its values. Non-data values like strings must stay static.

Take a Flax model: you would want JAX to only track and optimize its parameters, and the RNG keys. For trivial things like the model hyperparameters (e.g., the param shape), they can stay static to save compilation bandwidth and allow code path customization.

Current Flax module automatically classifies this for you. Only the `jax.Array` attributes are treated as dynamic data, unless you explicitly wrap a data value using `nnx.Variable` classes.


```python
class Foo(nnx.Module):
  def __init__(self, dim, rngs):
    self.w = nnx.Param(jax.random.normal(rngs.param(), (dim, dim)))
    self.dim = dim
    self.traced_dim = nnx.Param(dim)  # This became traced!
    self.rng = rngs

foo = Foo(4, nnx.Rngs(0))
for kp, x in jax.tree.flatten_with_path(nnx.state(foo))[0]:
  print(f'{jax.tree_util.keystr(kp)}: {x}')
```

    ['rng']['default']['count'].value: 1
    ['rng']['default']['key'].value: Array((), dtype=key<fry>) overlaying:
    [0 0]
    ['traced_dim'].value: 4
    ['w'].value: [[ 1.0040143  -0.9063372  -0.7481722  -1.1713669 ]
     [-0.8712328   0.5888381   0.72392994 -1.0255982 ]
     [ 1.661628   -1.8910251  -1.2889339   0.13360691]
     [-1.1530392   0.23929629  1.7448074   0.5050189 ]]


When compiling a function using this pytree, you'll notice the difference between traced and static values. You can only use static ones in control flows.


```python
gdef, state = nnx.split(foo)

@jax.jit
def jitted(state):
  model = nnx.merge(gdef, state)
  print(f'{model.dim = }')
  print(f'{model.traced_dim.value = }')  # This is being traced
  if model.dim == 4:
    print('Code path based on static data value works fine.')
  try:
    if model.traced_dim.value == 4:
      print('This will never run :(')
  except jax.errors.TracerBoolConversionError as e:
    print(f'Code path based on JAX data value throws error: {e}')

jitted(state)
```

    model.dim = 4
    model.traced_dim.value = Traced<~int32[]>with<DynamicJaxprTrace>
    Code path based on static data value works fine.
    Code path based on JAX data value throws error: Attempted boolean conversion of traced array with shape bool[].
    The error occurred while tracing the function jitted at /var/folders/4c/ylxxyg_n67957jf6616c7z5000gbn1/T/ipykernel_25111/4039468265.py:3 for jit. This concrete value was not available in Python because it depends on the value of the argument state['traced_dim'].value.
    See https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError


## Abstract arrays and pytrees

Abstract array is a JAX class to represent an array not by its value, but simply by its metadata information like shape, dtype and sharding. It is fast and handy because it doesn't allocate any memory for the array data.

You can construct an abstract array by calling [`jax.ShapeDtypeStruct`](https://docs.jax.dev/en/latest/_autosummary/jax.ShapeDtypeStruct.html) on your own, or use [`jax.eval_shape`](https://docs.jax.dev/en/latest/_autosummary/jax.eval_shape.html), which takes a function and arguments and returns the abstract version of its output.


```python
print(x)
abs_x = jax.eval_shape(lambda x: x, x)
print(abs_x)
```

    [[ 1.0040143  -0.9063372  -0.7481722  -1.1713669 ]
     [-0.8712328   0.5888381   0.72392994 -1.0255982 ]
     [ 1.661628   -1.8910251  -1.2889339   0.13360691]
     [-1.1530392   0.23929629  1.7448074   0.5050189 ]]
    ShapeDtypeStruct(shape=(4, 4), dtype=float32)


It is a good way to dry-run your code and debug a model without any actual compute and memory cost. For example, you can have an overview of the parameters inside this very large model.


```python
class MLP(nnx.Module):
  def __init__(self, dim, nlayers, rngs):
    self.blocks = [nnx.Linear(dim, dim, rngs=rngs) for _ in range(nlayers)]
    self.activation = jax.nn.relu
    self.nlayers = nlayers
  def __call__(self, x):
    for block in self.blocks:
      x = self.activation(block(x))
    return x

dim, nlayers = 8190, 64   # Some very big numbers
@partial(jax.jit, static_argnums=(0, 1))
def init_state(dim, nlayers):
  model = MLP(dim, nlayers, nnx.Rngs(0))
  return nnx.state(model)
abstract_state = jax.eval_shape(partial(init_state, dim, nlayers))
print(abstract_state['blocks'][0])
```

    [38;2;79;201;177mState[0m[38;2;255;213;3m({[0m[38;2;105;105;105m[0m
      [38;2;156;220;254m'bias'[0m[38;2;212;212;212m: [0m[38;2;79;201;177mVariableState[0m[38;2;255;213;3m([0m[38;2;105;105;105m # 8,190 (32.8 KB)[0m
        [38;2;156;220;254mtype[0m[38;2;212;212;212m=[0m[38;2;79;201;177mParam[0m,
        [38;2;156;220;254mvalue[0m[38;2;212;212;212m=[0mShapeDtypeStruct(shape=(8190,), dtype=float32)
      [38;2;255;213;3m)[0m,
      [38;2;156;220;254m'kernel'[0m[38;2;212;212;212m: [0m[38;2;79;201;177mVariableState[0m[38;2;255;213;3m([0m[38;2;105;105;105m # 67,076,100 (268.3 MB)[0m
        [38;2;156;220;254mtype[0m[38;2;212;212;212m=[0m[38;2;79;201;177mParam[0m,
        [38;2;156;220;254mvalue[0m[38;2;212;212;212m=[0mShapeDtypeStruct(shape=(8190, 8190), dtype=float32)
      [38;2;255;213;3m)[0m
    [38;2;255;213;3m})[0m


Once you have an abstract pytree for your function input or output, it's easier to describe how you want your data to be sharded. You should use such a pytree with sharding information to instruct your checkpoint loading library to load your arrays distributedly. Our checkpointing guide contains [an example of how to do this](https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html#load-a-sharded-model-from-a-checkpoint).

## Distributed computing

Another big use case for abstract pytrees is to tell JAX machinery how you want each array to be sharded during any point of your computation.

Remember what we mentioned earlier? JAX handles the actual computation and data allocation on accelerators. This means you **must** use some `jax.jit`-compiled function to run any distributed computation task. 

In most of the cases, this means you specify sharding to each of your parameters via JAX API. There are many ways to do this. The code below just briefly describes two ways - one with a `with_sharding_constraint` inside your jitted function, and another with `jax.jit`'s argument `out_shardings`.


```python
# Some smaller numbers so that we actually can run it
dim, nlayers = 1024, 8
abstract_state = jax.eval_shape(partial(init_state, dim, nlayers))
mesh = jax.make_mesh((jax.device_count(), ), 'model')

# Generate sharding for each of your params manually, sharded along the last axis.
def make_sharding(abs_x):
  P = jax.sharding.PartitionSpec
  pspec = P(None, 'model') if len(abs_x.shape) > 1 else P('model')
  return jax.sharding.NamedSharding(mesh, pspec)
state_shardings = jax.tree.map(make_sharding, abstract_state)
```


```python
# Option 1
@partial(jax.jit, static_argnums=(0, 1))
def init_sharded_state(dim, nlayers):
  model = MLP(dim, nlayers, nnx.Rngs(0))
  state = nnx.state(model)
  return jax.lax.with_sharding_constraint(state, state_shardings)
state = init_sharded_state(dim, nlayers)
jax.debug.visualize_array_sharding(state['blocks'][0]['kernel'].value)

# Option 2
@partial(jax.jit, static_argnums=(0, 1), out_shardings=state_shardings)
def init_sharded_state2(dim, nlayers):
  model = MLP(dim, nlayers, nnx.Rngs(0))
  return nnx.state(model)
state = init_sharded_state2(dim, nlayers)
jax.debug.visualize_array_sharding(state['blocks'][0]['kernel'].value)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">  CPU 0  </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">  CPU 1  </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">  CPU 2  </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">  CPU 3  </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">  CPU 4  </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">  CPU 5  </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">  CPU 6  </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">  CPU 7  </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">  CPU 0  </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">  CPU 1  </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">  CPU 2  </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">  CPU 3  </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">  CPU 4  </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">  CPU 5  </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">  CPU 6  </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">  CPU 7  </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">         </span><span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">         </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">         </span>
</pre>



The example below are just to showcase how to do sharding in pure JAX API. Flax offers a small API to annotate the sharding when you define a parameter, so that you don't have to write an arbitrary `make_sharding()` function at top level. Check out our [GSPMD guide](https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html) to learn more.

## Transformations

For Flax transforms and their relation with JAX transforms, refer to [Flax Transforms](https://flax.readthedocs.io/en/latest/guides/transforms.html) doc. The usage of transforms will greatly change after Flax modules become pytrees, because directly using JAX transforms will be possible.
