# FLIP: Axis Metadata


- Start Date: 2022-08-08
- FLIP Issue: [#2434](https://github.com/google/flax/issues/2434)
- FLIP PR: [#2435](https://github.com/google/flax/pull/2435)
- Status: Proposal


## Summary

This FLIP proposes to extend Flax's variable collections with a generic axis metadata API. 
The core of the API is a abstract base class that is recognized by lifting transformations that can add an axis (vmap, scan).
Users can extend the base class to keep track of per-axis metadata in a way that works with lifted transformations.


## Motivation

Generally, there is no way in Flax to track metadata for variables across lifted transformations. Axis metadata is used to keep track of semantic information about axes into other (Flax independent) APIs.
For example, optimizers like AdaFactor can be configured on a per-axis level and partitioning APIs in JAX like xmap or pjit require per variable annotations to map effectiently to parallel hardware.

Currently, there is experimental support for partitioning annotations which requires using dedicated wrapper around lifted transforms that change axes (``nn.scan``, ``nn.vmap``) and a special APIs to create variables (``param_with_axes`` and ``variable_with_axes``). 
The experimental partitioning API stores the metadata in a seperate collection named "[collection]_axes".

The experimental API has a number of shortcomings that we like to solve:
1. The current API works for tracking PartitionSpecs but not for other types of metadata like optimizer annotations.
2. The implementation using an "xxx_axes" collection requires error-prone and non-composable string manipulation.
3. Special, partioning-aware variable creators and lifted transforms are required
4. The partioning API is hard to use with pre-existing Modules that aren't partioning aware.


## Proposal

To generalize metadata tracking and keep the specific metadata out of core Flax we propose the following abstract base class:

```python
class AxisMetadata(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def unbox(self) -> Any:
    pass

  @abc.abstractmethod
  def add_axis(self, index: int, params: Dict[Any, Any]) -> TAxisMetadata:
    pass

  @abc.abstractmethod
  def remove_axis(self, index: int, params: Dict[Any, Any]) -> TAxisMetadata:
    pass
```

We call this type of class wrapping a value and keeping track of some additional data a **box**.
By defining an abstract base class for this box, the API does not need to be aware of the specifics of the metadata that is tracked.
This should make the API future proof and modular.

The ``add_axis`` and ``remove_axis`` callback return an instance of their own type instead of mutating in-place.
Typically, an implementation would be a ``flax.struct.PyTreeNode`` because the box should still be a valid JAX value and must therefore be handled by the PyTree API.
Calling ``jax.tree_map`` on a boxed value will simply map over the value in the box.
The lifted transforms that need to handle metadata will call ``jax.tree_map(..., is_leaf=lambda x: isinstance(x, AxisMetadata))`` to find the AxisMetadata instances within a PyTree.

Advantages of the boxing approach:
1. Boxing can be used outside of Flax and metadata is automatically "inherited". For example, the compiler state will
   have the same partitioning spec as the parameters, because the state is initialized using a ``jax.tree_map`` over the boxed parameters.
2. Boxes are composable.
3. Boxing avoids string manipulation and generally avoids having to handle additional auxilary collections like "param_axes" in the current
   partitioning API.
4. No need to lift metadata collections seperately.


Disadvantages:
1. Handling boxed values requires the relatively new ``is_leaf=`` syntax which users might not be familiar with. Although users will
   probably call Flax provided utils that handle teh low-level tree_map calls in most cases.
3. Custom Pytree nodes have a small runtime overhead. It's hard to observe this in practise because JAX calls are async.


### Init syntax


Boxes can be created directly by the init function of a variable. Therefore, we propose to create metadata using higher-order initializers.
The main advantage of this is that we can decouple metadata handling completly from the Module definition. Also, most Modules already over
attributes to override the default initialzers so users can add metadata to existing Modules without requiring any code changes.

Too illustrate this, let's consider a metadata class that keeps track of PartitionSpecs used by ``pjit``:

```python
class Partitioned(flax.struct.PyTreeNode, AxisMetadata):
  value: Any
  named_axes: Tuple[Optional[str]]
  
  ...

def with_partitioning(init_fn, names):
  def wrapper(*args, **kwargs):
    return Partitioned(init_fn(*args, **kwargs), names)
  return wrapper
```

Here we also defined a small utility called ``with_partitioning`` that we can use to wrap existing initialzers to add metadata: 


```python
# init kernel with lecun normal and split the output features over the data axis
partitioned_dense = nn.Dense(features, kernel_init=with_partitioning(nn.initializers.lecun_normal, [None, "data"]))
```


### Unbox syntax


Metadata typically doesn't need to be handled by Modules directly. Therefore, we prosose to make Modules agnostic to Metadata boxes by default.
The ``unbox`` method can be used to unpack a variable such that only the original JAX arrays remain. Users can manually call unbox but to make
sure Module classes don't have to call it everywhere we add an unbox keyword arg to variable returning APIs (e.g.: ``.param``, ``.variable``, ``.get_variable``).
The keyword arg ``unbox`` will default to ``True`` such that a Modules are metadata agnostic by default. This also means existing Modules will be backward compatible
with the new API.

```python
kernel = self.param("kernel", self.kernel_init, shape)  # No AxisMetadata instances
kernel_box = self.get_variable("param", "kernel", unbox=False)  # AxisMetadata boxes are preserved
```


### Lift syntax

When calling a lifted transformation that adds an axis you will now be able to pass a dictionary with arguments.
These params will be passed to ``AxisMetadata`` add_axis/remove_axis callbacks:

```python
nn.scan(..., variable_axes={"params": 0}, metadata_params={nn.Partitioned.AXIS_NAME: "layers"})
```

A dict is used such that users can add their own arguments to custom AxisMetadata classes.

