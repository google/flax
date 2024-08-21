# Model surgery

In this guide you will learn how to do model surgery with Flax NNX with several real-scenario use cases:

* __Pythonic module manipulation__: Pythonic ways to manipulate sub-modules given a model.

* __Manipulating an abstract model or state__: A key trick to play with Flax NNX modules and states without memory allocation.

* __Checkpoint surgery: From a raw state to model__: How to manipulate parameter states when they are incompatible with existing model code.

* __Partial initialization__: How to initialize only a part of the model from scratch using a naive method or a memory-efficient method.


```python
from typing import *
from pprint import pprint
import functools

import jax
from jax import lax, numpy as jnp, tree_util as jtu

from jax.sharding import PartitionSpec, Mesh, NamedSharding
from jax.experimental import mesh_utils
import flax
from flax import nnx
import flax.traverse_util
import numpy as np
import orbax.checkpoint as orbax

key = jax.random.key(0)
```


```python
class TwoLayerMLP(nnx.Module):
  def __init__(self, dim, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(dim, dim, rngs=rngs)
    self.linear2 = nnx.Linear(dim, dim, rngs=rngs)

  def __call__(self, x):
    x = self.linear1(x)
    return self.linear2(x)
```

## Pythonic module manipulation

Doing model surgery is easiest when you already have a fully fleshed-out model loaded with correct parameters, and you don't intend to change your model definition code.

You can perform a variety of Pythonic operations on its sub-modules, such as sub-module swapping, module sharing, variable sharing, and monkey-patching:


```python
model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(42), (3, 4))
np.testing.assert_allclose(model(x), model.linear2(model.linear1(x)))

# Sub-module swapping
original1, original2 = model.linear1, model.linear2
model.linear1, model.linear2 = model.linear2, model.linear1
np.testing.assert_allclose(model(x), original1(original2(x)))

# Module sharing (tying all weights)
model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
model.linear2 = model.linear1
assert not hasattr(nnx.state(model), 'linear2')
np.testing.assert_allclose(model(x), model.linear1(model.linear1(x)))

# Variable sharing (weight-tying)
model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
model.linear1.kernel = model.linear2.kernel  # the bias parameter is kept separate
assert hasattr(nnx.state(model), 'linear2')
assert hasattr(nnx.state(model)['linear2'], 'bias')
assert not hasattr(nnx.state(model)['linear2'], 'kernel')

# Monkey-patching
model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
def awesome_layer(x): return x
model.linear2 = awesome_layer
np.testing.assert_allclose(model(x), model.linear1(x))

```

## Creating an abstract model or state without memory allocation

For more complex model surgery, a key technique is creating and manipulating an abstract model or state without allocating any real parameter data. This makes trial iteration faster and removes any concern on memory constraints.

To create an abstract model,
* Create a function that returns a valid NNX model; and
* Run `nnx.eval_shape` (not `jax.eval_shape`) upon it.

Now you can use `nnx.split` as usual to get its abstract state. Note that all the fields that should be `jax.Array` in a real model are now an abstract `jax.ShapeDtypeStruct` with only shape/dtype/sharding information.


```python
abs_model = nnx.eval_shape(lambda: TwoLayerMLP(4, rngs=nnx.Rngs(0)))
gdef, abs_state = nnx.split(abs_model)
pprint(abs_state)
```

    State({
      'linear1': {
        'bias': VariableState(
          type=Param,
          value=ShapeDtypeStruct(shape=(4,), dtype=float32)
        ),
        'kernel': VariableState(
          type=Param,
          value=ShapeDtypeStruct(shape=(4, 4), dtype=float32)
        )
      },
      'linear2': {
        'bias': VariableState(
          type=Param,
          value=ShapeDtypeStruct(shape=(4,), dtype=float32)
        ),
        'kernel': VariableState(
          type=Param,
          value=ShapeDtypeStruct(shape=(4, 4), dtype=float32)
        )
      }
    })


When you fill every `VariableState` leaf's `value`s with real jax arrays, the abstract model becomes equivalent to a real model.


```python
model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
abs_state['linear1']['kernel'].value = model.linear1.kernel
abs_state['linear1']['bias'].value = model.linear1.bias
abs_state['linear2']['kernel'].value = model.linear2.kernel
abs_state['linear2']['bias'].value = model.linear2.bias
nnx.update(abs_model, abs_state)
np.testing.assert_allclose(abs_model(x), model(x))  # They are equivalent now!
```

## Checkpoint surgery

With the abstract state technique in hand, you can do arbitrary manipulation on any checkpoint (or runtime parameter pytree) to make them fit with your given model code, then call `nnx.update` to merge them.

This can be helpful when you are trying to change model code significantly (for example, when migrating from Flax Linen to Flax NNX), and old weights are no longer naturally compatible. Let's run a simple example here:


```python
# Save a version of a model into a checkpoint.
checkpointer = orbax.PyTreeCheckpointer()
old_model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
checkpointer.save(f'/tmp/nnx-surgery-state', nnx.state(model), force=True)
```

In this new model, the sub-modules are renamed from `linear(1|2)` to `layer(1|2)`. Since the pytree structure changed, it's impossible to load the old checkpoint with the new model state structure:


```python
class ModifiedTwoLayerMLP(nnx.Module):
  def __init__(self, dim, rngs: nnx.Rngs):
    self.layer1 = nnx.Linear(dim, dim, rngs=rngs)  # no longer linear1!
    self.layer2 = nnx.Linear(dim, dim, rngs=rngs)

  def __call__(self, x):
    x = self.layer1(x)
    return self.layer2(x)

abs_model = nnx.eval_shape(lambda: ModifiedTwoLayerMLP(4, rngs=nnx.Rngs(0)))
try:
  with_item = checkpointer.restore('/tmp/nnx-surgery-state', item=nnx.state(abs_model))
  print(with_item)
except Exception as e:
  print(f'This will throw error: {type(e)}: {e}')
```

    This will throw error: <class 'KeyError'>: 'layer1'


    /Users/ivyzheng/envs/py310/lib/python3.10/site-packages/orbax/checkpoint/type_handlers.py:1401: UserWarning: Couldn't find sharding info under RestoreArgs. Populating sharding info from sharding file. Please note restoration time will be slightly increased due to reading from file instead of directly from RestoreArgs. Note also that this option is unsafe when restoring on a different topology than the checkpoint was saved with.
      warnings.warn(


But you can load the parameter tree as a raw dictionary, make the renames, and generate a new state that is guaranteed to be compatible with your new model definition.


```python
def module_from_variables_dict(module_factory, variables, map_key_fn):
  if map_key_fn is None:
    map_key_fn = lambda path: path
  mdl = nnx.eval_shape(module_factory)
  graph_def, state = nnx.split(mdl)
  state = state.flat_state()
  for path, val in flax.traverse_util.flatten_dict(variables).items():
    mapped_path = map_key_fn(path)
    if mapped_path not in state:
      raise ValueError(f"{mapped_path} doesn't exist in {state.keys()}")
    state[mapped_path].value = val
  state = nnx.State.from_flat_path(state)
  return nnx.merge(graph_def, state)

# Make your local change on the checkpoint.
raw = checkpointer.restore('/tmp/nnx-surgery-state')
pprint(raw)
raw['layer1'], raw['layer2'] = raw['linear1'], raw['linear2']
del raw['linear1'], raw['linear2']

restored_model = module_from_variables_dict(
  lambda: nnx.eval_shape(lambda: ModifiedTwoLayerMLP(4, rngs=nnx.Rngs(0))),
  raw,
  lambda path: path[:-1] if path[-1] == 'raw_value' else path
)

np.testing.assert_allclose(restored_model(jnp.ones((3, 4))), old_model(jnp.ones((3, 4))))
```

    {'linear1': {'bias': {'raw_value': Array([0., 0., 0., 0.], dtype=float32)},
                 'kernel': {'raw_value': Array([[-0.80345297, -0.34071913, -0.9408296 ,  0.01005968],
           [ 0.26146442,  1.1247735 ,  0.54563737, -0.374164  ],
           [ 1.0281805 , -0.6798804 , -0.1488401 ,  0.05694951],
           [-0.44308168, -0.60587114,  0.434087  , -0.40541083]],      dtype=float32)}},
     'linear2': {'bias': {'raw_value': Array([0., 0., 0., 0.], dtype=float32)},
                 'kernel': {'raw_value': Array([[ 0.21010089,  0.8289361 ,  0.04589564,  0.5422644 ],
           [ 0.41914317,  0.84359694, -0.47937787, -0.49135214],
           [-0.46072108,  0.4630125 ,  0.39276958, -0.9441406 ],
           [-0.6690758 , -0.18474789, -0.57622856,  0.4821079 ]],      dtype=float32)}}}


## Partial initialization

In some cases (such as with LoRA), you may want to randomly-initialize only *part of* your model parameters.  This can be achieved through naive partial initialization or memory-efficient partial initialization.

### Naive partial initialization

You can simply initialize the whole model, then swap pre-trained parameters in. But this approach could allocate additional memory midway, if your modification requires re-creating module parameters that you will later discard. See this example below.

> Note: You can use `jax.live_arrays()` to check all the arrays live in memory at any given time. This call can be messed up when you run a single notebook cell multiple times (due to garbage-collecting old python variables), but restarting the kernel and running from scratch will always yield same output.


```python
# Some pretrained model state.
old_state = nnx.state(TwoLayerMLP(4, rngs=nnx.Rngs(0)))

simple_model = nnx.eval_shape(lambda: TwoLayerMLP(4, rngs=nnx.Rngs(42)))
print(f'Number of jax arrays in memory at start: {len(jax.live_arrays())}')
# On this line, extra kernel and bias is created inside the new LoRALinear!
# They are wasted since you are going to use the kernel and bias in `old_state` anyway.
simple_model.linear1 = nnx.LoRALinear(4, 4, lora_rank=3, rngs=nnx.Rngs(42))
print(f'Number of jax arrays in memory midway: {len(jax.live_arrays())}'
      ' (4 new created in LoRALinear - kernel, bias, lora_a & lora_b)')
nnx.update(simple_model, old_state)
print(f'Number of jax arrays in memory at end: {len(jax.live_arrays())}'
      ' (2 discarded - only lora_a & lora_b are used in model)')
```

    Number of jax arrays in memory at start: 34
    Number of jax arrays in memory midway: 38 (4 new created in LoRALinear - kernel, bias, lora_a & lora_b)
    Number of jax arrays in memory at end: 36 (2 discarded - only lora_a & lora_b are used in model)


### Memory-efficient partial initialization

Use `nnx.jit`'s efficiently compiled code to make sure only the state parameters you need are initialized:


```python
# Some pretrained model state
old_state = nnx.state(TwoLayerMLP(4, rngs=nnx.Rngs(0)))

# Use `nnx.jit` (which wraps `jax.jit`) to automatically skip unused arrays - memory efficient!
@functools.partial(nnx.jit, donate_argnums=0, static_argnums=1)
def partial_init(old_state, rngs):
  model = TwoLayerMLP(4, rngs=rngs)
  # Create a new state.
  model.linear1 = nnx.LoRALinear(4, 4, lora_rank=3, rngs=rngs)
  # Add the existing state.
  nnx.update(model, old_state)
  return model

print(f'Number of jax arrays in memory at start: {len(jax.live_arrays())}')
# Note that `old_state` will be deleted after this `partial_init` call.
good_model = partial_init(old_state, nnx.Rngs(42))
print(f'Number of jax arrays in memory at end: {len(jax.live_arrays())}'
      ' (2 new created - lora_a and lora_b)')
```

    Number of jax arrays in memory at start: 40
    Number of jax arrays in memory at end: 42 (2 new created - lora_a and lora_b)

