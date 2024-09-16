# FLIP: kw_only dataclasses
Authors: Brennan Saeta, Ivy Zheng

 - Start Date: Mar 23, 2023
 - FLIP Issue: [TBD]
 - FLIP PR: #2974
 - Status: Implementing


## Summary

Python 3.10 adds support for `kw_only` dataclasses. Subclasses of `flax.linen.Module` are automatically converted to `dataclasses` on users' behalf, but today, Flax doesn't allow setting the `kw_only` parameter to this dataclass transform, even if users are running Python 3.10. This proposal allows users to use this new feature with `nn.Module`'s.


## Motivation

In larger Flax-based codebases (e.g. [`PaxML`](https://github.com/google/paxml) / [`Praxis`](https://github.com/google/praxis)), it’s not uncommon to define an (abstract) subclass of nn.Module that contains shared functionality that is itself further subclassed for specific implementations (e.g. [`BaseLayer`](https://github.com/google/praxis/blob/main/praxis/base_layer.py), or [`StackedTransformerRepeat`](https://github.com/google/praxis/blob/81479b260fcc13de8549cdbfb0fdf5c3f188ac90/praxis/layers/transformers.py#L1836) which is further subclassed by [`PipelineCompatibleStackedTransformerRepeat`](https://github.com/google/praxis/blob/81479b260fcc13de8549cdbfb0fdf5c3f188ac90/praxis/layers/transformers.py#L2198)).

Often, these parent types define hyperparameters (constructor arguments), often with default values. Without `kw_only` on the `dataclass` transform, default values must be specified for all child layers hyperparameters. This is suboptimal, because users could forget to set them when instantiating the modules. For example, `Child` must set a default value for `num_heads` (because a non-defaulted argument can’t come after a defaulted argument if they are positional), but no reasonable default is available:

```python
class BaseLayer(nn.Module):
  mesh: Optional[jax.experimental.mesh.Mesh] = None

  def with_sharding(self, some_variable, some_sharding):
    if self.mesh:
      # Do something useful here.

class Child(BaseLayer):
  num_heads: int  # Don't want to have to set a default argument!

  def __call__(self, x):
    ...
```

Note: Flax already has this problem, which is why `nn.Module` has its own fancy `kw_only_dataclasses.dataclass` transform: it moves the `name` and `parent` dataclass fields to the end, so they can have defaults.


## Implementation

To allow modules to optionally opt into this `kw_only` dataclass behavior, we leverage arguments to `__init_subclass__`. This would look as follows:

```python
class BaseLayer(nn.Module, kw_only=True):
  ...

class Child(BaseLayer):
  ...
```

The implementation of `nn.Module`’s `__init_subclass__` will be tweaked as follows:

```python
class Module(ModuleBase):
  def __init_subclass__(self, kw_only: Optional[bool] = None):
    # ...
    if kw_only:
     if is_python_310_or_above():
       dataclass_transform_args = {'kw_only': True}
     else:
       raise TypeError("Can't use `kw_only` before Py3.10.")
    else:
       dataclass_transform_args = {}

    kw_only_dataclasses.dataclass(
      cls, unsafe_hash='__hash__' not in cls.__dict__,
      repr=False,
      **dataclass_transform_args)
```

### Forward compatibility

For future simplification, if `kw_only` is requested and the Python version is 3.10 or above, bypass the `kw_only_dataclasses` implementation and just use the regular `dataclasses` transform.

That means we may one day remove `flax/linen/kw_only_dataclasses.py` when Flax rolls over 3.10.


## Discussion

### Aligned with Python `dataclass`

We prefer to keep the behavior of `nn.Module`’s `kw_only` aligned with the Python dataclasses. Note that this means `kw_only` will not be inheritable, and this could happen:

```python
class BaseLayer(nn.Module, kw_only=True):
  base_muliplier: Optional[int] = -1

class ChildLayer(BaseLayer):
  child_multiplier: int

BaseLayer(2)   # This will throw error
ChildLayer(2)  # But this will not
```

### `flax.struct.dataclass`

There’s a potentially related feature to allow `kw_only` to be specified for `flax.struct.dataclass`. This should be considered an orthogonal decision.


