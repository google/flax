---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Model surgery

Model surgery is an act of making modifications on an existing neural network's building blocks and parameters, such as layer replacement, parameter or state manipulation, or even "monkey patching". In this guide, you will learn how to perform model surgery in Flax NNX using several real-world scenarios:

* __Pythonic `nnx.Module` manipulation__: Using Pythonic ways to manipulate sub-`Module`s given a model.

* __Manipulation of an abstract model or state__: A key trick for playing with `flax.nnx.Module`s and states without memory allocation.

* __Checkpoint surgery from a raw state to model__: How to manipulate parameter states when they are incompatible with existing model code.

* __Partial initialization__: How to initialize only a part of the model from scratch using a naive method or a memory-efficient method.

```{code-cell} ipython3
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

```{code-cell} ipython3
class TwoLayerMLP(nnx.Module):
  def __init__(self, dim, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(dim, dim, rngs=rngs)
    self.linear2 = nnx.Linear(dim, dim, rngs=rngs)

  def __call__(self, x):
    x = self.linear1(x)
    return self.linear2(x)
```

## Pythonic `nnx.Module` manipulation

It is easier to perform model surgery when:

1) You already have a fully fleshed-out model loaded with correct parameters; and
2) You don't intend to change your model definition code.

You can perform a variety of Pythonic operations on its sub-`Module`s, such as sub-`Module` swapping, `Module` sharing, variable sharing, and monkey-patching:

```{code-cell} ipython3
model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.key(42), (3, 4))
np.testing.assert_allclose(model(x), model.linear2(model.linear1(x)))

# Sub-`Module` swapping.
original1, original2 = model.linear1, model.linear2
model.linear1, model.linear2 = model.linear2, model.linear1
np.testing.assert_allclose(model(x), original1(original2(x)))

# `Module` sharing (tying all weights together).
model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
model.linear2 = model.linear1
assert not hasattr(nnx.state(model), 'linear2')
np.testing.assert_allclose(model(x), model.linear1(model.linear1(x)))

# Variable sharing (weight-tying).
model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
model.linear1.kernel = model.linear2.kernel  # the bias parameter is kept separate
assert 'linear2' in nnx.state(model)
assert 'bias' in nnx.state(model)['linear2']
assert not hasattr(nnx.state(model)['linear2'], 'kernel')

# Monkey-patching.
model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
def awesome_layer(x): return x
model.linear2 = awesome_layer
np.testing.assert_allclose(model(x), model.linear1(x))
```

## Creating an abstract model or state without memory allocation

To do more complex model surgery, the key technique you can use is creating and manipulating an abstract model or state without allocating any real parameter data. This makes trial iteration faster and removes any concern on memory constraints.

To create an abstract model:

* Create a function that returns a valid Flax NNX model; and
* Run `nnx.eval_shape` (not `jax.eval_shape`) upon it.

Now you can use `nnx.split` as usual to get its abstract state. Note that all fields that should be `jax.Array`s in a real model are now of an abstract `jax.ShapeDtypeStruct` type with only shape/dtype/sharding information.

```{code-cell} ipython3
abs_model = nnx.eval_shape(lambda: TwoLayerMLP(4, rngs=nnx.Rngs(0)))
gdef, abs_state = nnx.split(abs_model)
pprint(abs_state)
```

When you fill every `nnx.VariableState` pytree leaf's `value` attributes with real `jax.Array`s, the abstract model becomes equivalent to a real model.

```{code-cell} ipython3
model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
abs_state['linear1']['kernel'].value = model.linear1.kernel
abs_state['linear1']['bias'].value = model.linear1.bias
abs_state['linear2']['kernel'].value = model.linear2.kernel
abs_state['linear2']['bias'].value = model.linear2.bias
nnx.update(abs_model, abs_state)
np.testing.assert_allclose(abs_model(x), model(x))  # They are equivalent now!
```

## Checkpoint surgery

With the abstract state technique in hand, you can perform arbitrary manipulation on any checkpoint - or runtime parameter pytree - to make them fit with your given model code, and then call `nnx.update` to merge them.

This can be helpful if you are trying to significantly change the model code - for example, when migrating from Flax Linen to Flax NNX - and old weights are no longer naturally compatible.

Let's run a simple example here:

```{code-cell} ipython3
# Save a version of model into a checkpoint
checkpointer = orbax.PyTreeCheckpointer()
old_model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
checkpointer.save(f'/tmp/nnx-surgery-state', nnx.state(model), force=True)
```

In this new model, the sub-`Module`s are renamed from `linear(1|2)` to `layer(1|2)`. Since the pytree structure has changed, it is impossible to directly load the old checkpoint with the new model state structure:

```{code-cell} ipython3
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

However, you can load the parameter pytree as a raw dictionary, perform the renames, and generate a new state that is guaranteed to be compatible with your new model definition.

```{code-cell} ipython3
def process_raw_dict(raw_state_dict):
  flattened = nnx.traversals.flatten_mapping(raw_state_dict)
  # Cut the '.value' postfix on every leaf path.
  flattened = {(path[:-1] if path[-1] == 'value' else path): value
               for path, value in flattened.items()}
  return nnx.traversals.unflatten_mapping(flattened)

# Make your local change on the checkpoint dictionary.
raw_dict = checkpointer.restore('/tmp/nnx-surgery-state')
pprint(raw_dict)
raw_dict['layer1'] = raw_dict.pop('linear1')
raw_dict['layer2'] = raw_dict.pop('linear2')

# Fit it into the model state.
abs_model = nnx.eval_shape(lambda: ModifiedTwoLayerMLP(4, rngs=nnx.Rngs(0)))
graph_def, state = nnx.split(abs_model)
nnx.replace_by_pure_dict(state, process_raw_dict(raw_dict))
restored_model = nnx.merge(graph_def, state)

np.testing.assert_allclose(restored_model(jnp.ones((3, 4))), old_model(jnp.ones((3, 4))))
```

## Partial initialization

In some cases - such as with LoRA (Low-Rank Adaption) - you may want to randomly-initialize only *part of* your model parameters. This can be achieved through:

- Naive partial initialization; or
- Memory-efficient partial initialization.

+++

### Naive partial initialization

To do naive partial initialization, you can just initialize the whole model, then swap the pre-trained parameters in. However, this approach may allocate additional memory midway if your modification requires re-creating module parameters that you will later discard. Below is an example of this.

> **Note:** You can use `jax.live_arrays()` to check all the arrays live in memory at any given time. This call can be “messed up” when you run a single Jupyter notebook cell multiple times (due to garbage-collection of old Python variables). However, restarting the Python kernel in the notebook and running the code from scratch will always yield the same output.

```{code-cell} ipython3
# Some pretrained model state
old_state = nnx.state(TwoLayerMLP(4, rngs=nnx.Rngs(0)))

simple_model = nnx.eval_shape(lambda: TwoLayerMLP(4, rngs=nnx.Rngs(42)))
print(f'Number of jax arrays in memory at start: {len(jax.live_arrays())}')
# In this line, extra kernel and bias is created inside the new LoRALinear!
# They are wasted, because you are going to use the kernel and bias in `old_state` anyway.
simple_model.linear1 = nnx.LoRALinear(4, 4, lora_rank=3, rngs=nnx.Rngs(42))
print(f'Number of jax arrays in memory midway: {len(jax.live_arrays())}'
      ' (4 new created in LoRALinear - kernel, bias, lora_a & lora_b)')
nnx.update(simple_model, old_state)
print(f'Number of jax arrays in memory at end: {len(jax.live_arrays())}'
      ' (2 discarded - only lora_a & lora_b are used in model)')
```

### Memory-efficient partial initialization

To do memory-efficient partial initialization, use `nnx.jit`'s efficiently compiled code to make sure only the state parameters you need are initialized:

```{code-cell} ipython3
# Some pretrained model state
old_state = nnx.state(TwoLayerMLP(4, rngs=nnx.Rngs(0)))

# Use `nnx.jit` (which wraps `jax.jit`) to automatically skip unused arrays - memory efficient!
@nnx.jit(donate_argnums=0)
def partial_init(old_state, rngs):
  model = TwoLayerMLP(4, rngs=rngs)
  # Create a new state.
  model.linear1 = nnx.LoRALinear(4, 4, lora_rank=3, rngs=rngs)
  # Add the existing state.
  nnx.update(model, old_state)
  return model

print(f'Number of JAX Arrays in memory at start: {len(jax.live_arrays())}')
# Note that `old_state` will be deleted after this `partial_init` call.
good_model = partial_init(old_state, nnx.Rngs(42))
print(f'Number of JAX Arrays in memory at end: {len(jax.live_arrays())}'
      ' (2 new created - lora_a and lora_b)')
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
