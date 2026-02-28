# FLIP: Axis Metadata


- Start Date: 2022-08-08
- FLIP Issue: [#2434](https://github.com/google/flax/issues/2434)
- FLIP PR: [#2435](https://github.com/google/flax/pull/2435)
- Status: Proposal


## Summary

This FLIP proposes to extend Flax's variable collections with a generic axis metadata API.
The core of the API is an abstract base class that is recognized by lifting transformations that can add an axis (vmap, scan).
Users can extend the base class to keep track of per-axis metadata in a way that works with lifted transformations.


## Motivation

Generally, there is no way in Flax to track metadata for variables across lifted transformations.
Axis metadata is used to keep track of semantic information about axes into other (Flax independent) APIs.
For example, optimizers like AdaFactor can be configured on a per-axis level and partitioning APIs
in JAX like xmap or pjit require per variable annotations to map effectiently to parallel hardware.

Currently, there is an experimental [API](https://github.com/google/flax/blob/main/flax/linen/partitioning.py)
supporting partitioning annotations with wrappers around lifted transforms that change axes (``nn.scan_with_axes``, ``nn.vmap_with_axes``)
and a special APIs to create variables (``param_with_axes`` and ``variable_with_axes``).
The experimental partitioning API stores the metadata in a separate collection named "[collection]_axes".


The experimental API has a number of shortcomings that we like to solve:
1. The current API works for tracking PartitionSpecs but not for other types of metadata like optimizer annotations.
2. The implementation using an "xxx_axes" collection requires error-prone and non-composable string manipulation.
3. Special, partioning-aware variable creators and lifted transforms are required
4. The partioning API is hard to use with pre-existing Modules that aren't partioning aware.


## Proposal

To generalize metadata tracking and keep the specific metadata out of core Flax we propose the following abstract base class:

```python
TAxisMetadata = TypeVar("TAxisMetadata", bound="AxisMetadata")

class AxisMetadata(metaclass=abc.ABCMeta):
  """Abstract base class for boxed Metadata.

  ``AxisMetadata`` enables arbitrary, per axis metadata for variables.
  By using ``unbox`` the metadata is stripped away to obtain the original
  variables. By using unboxing, most code handling variables does not need
  to handle ``AxisMetadata`` specifically, but can directly operate on the JAX
  arrays that they wrap.

  Additionally, ``AxisMetadata`` supports updating metadata whenever an axis
  is added or removed by a functional transformation
  (e.g.: ``nn.scan`` or ``nn.vmap``) using the ``add_axis`` and ``remove_axis``
  methods.

  By extending ``AxisMetadata``, custom metadata can be stored. See
  ``Partitioned`` for a specific implementation.
  """

  @abc.abstractmethod
  def unbox(self) -> Any:
    """Returns the content of the AxisMetadata box.

    Note that unlike ``meta.unbox`` the unbox call should recursively unbox
    metadata. It should simply return value that it wraps directly even
    if that value itself is an instance of AxisMetadata.

    In practise, AxisMetadata subclasses should be registered as PyTree nodes to
    support passing instances to JAX and Flax APIs. The leaves returned for this
    note should correspond to the value returned by unbox.

    Returns:
      The unboxed value.
    """
    pass

  @abc.abstractmethod
  def add_axis(self: TAxisMetadata, index: int,
               params: Dict[Any, Any]) -> TAxisMetadata:
    """Adds a new axis to the axis metadata.

    Note that add_axis and remove_axis should act as each other's inverse
    (meaning: ``x.add_axis(i, p).remove_axis(i, p) == x``)

    Args:
      index: The position at which the new axis will be inserted
      params: An arbitrary dictionary of parameters passed by the transformation
        that introduces the new axis (e.g.: ``nn.scan`` or ``nn.vmap``). The
        user passes this dictionary as the `metadata_param` argument to the
        transformation.
    Returns:
      A new instance of the same type as self and with the same ``unbox``
      content with updated axis metadata.
    """
    pass

  @abc.abstractmethod
  def remove_axis(self: TAxisMetadata, index: int,
                  params: Dict[Any, Any]) -> TAxisMetadata:
    """Removes an axis from the axis metadata.

    Note that add_axis and remove_axis should act as each other's inverse
    (meaning: ``x.remove_axis(i, p).add_axis(i, p) == x``)

    Args:
      index: The position of the axis that is to be removed
      params: An arbitrary dictionary of parameters passed by the transformation
        that introduced the axis (e.g.: ``nn.scan`` or ``nn.vmap``). The
        user passes this dictionary as the `metadata_param` argument to the
        transformation.
    Returns:
      A new instance of the same type as self and with the same ``unbox``
      content with updated axis metadata.
    """
    pass
```

We call this type of class wrapping a value and keeping track of some additional data a **box**.
By defining an abstract base class for this box, the API does not need to be aware of the specifics of the metadata that is tracked.
This should make the API future proof and modular.

The ``add_axis`` and ``remove_axis`` method return an instance of their own type instead of mutating in-place.
Typically, an implementation would be a ``flax.struct.PyTreeNode`` because the box should still be a valid JAX value and must therefore be handled by the PyTree API.
Calling ``jax.tree.map`` on a boxed value will simply map over the value in the box.
The lifted transforms that need to handle metadata will call ``jax.tree.map(..., is_leaf=lambda x: isinstance(x, AxisMetadata))`` to find the AxisMetadata instances within a PyTree.

Advantages of the boxing approach:
1. Boxing can be used outside of Flax and metadata is automatically "inherited". For example, the optimizer state will
   have the same partitioning spec as the parameters, because the state is initialized using a ``jax.tree.map`` over the boxed parameters.
2. Boxes are composable.
3. Boxing avoids string manipulation and generally avoids having to handle additional auxiliary collections like "param_axes" in the current
   partitioning API.
4. No need to lift metadata collections separately.


Disadvantages:
1. Adding the boxes changes the PyTree hierarchy and introduces dataclasses within the otherwise plain, nested dict of variables.
3. Custom Pytree nodes have a small runtime overhead. It's hard to observe this in practise because JAX calls are async.


### Init syntax


Boxes can be created directly by the init function of a variable. Therefore, we propose to create metadata using higher-order initializers.
The main advantage of this is that we can decouple metadata handling completely from the Module definition. Also, most Modules already overwrite
attributes to override the default initialzers so users can add metadata to existing Modules without requiring any code changes.

To illustrate this, let's consider a metadata class that keeps track of PartitionSpecs used by ``pjit``:

```python
class Partitioned(flax.struct.PyTreeNode, AxisMetadata):
  value: Any
  names: Tuple[Optional[str], ...] = flax.struct.field(pytree_node=False)

  def add_axis(self, index: int, params: Dict[Any, Any]) -> TAxisMetadata:
    axis_name = self._get_partition_name(params)
    names = list(self.names)
    names.insert(index, axis_name)
    return self.replace(names=tuple(names))

  def remove_axis(self, index: int, params: Dict[Any, Any]) -> TAxisMetadata:
    axis_name = self._get_partition_name(params)
    names = list(self.names)
    assert names.pop(index) == axis_name
    return self.replace(names=tuple(names))

def with_partitioning(init_fn, names):
  def wrapper(*args, **kwargs):
    return Partitioned(init_fn(*args, **kwargs), names)
  return wrapper
```

Here we also defined a small utility called ``with_partitioning`` that we can use to wrap existing initialzers to add metadata:


```python
# init kernel with lecun normal and split the output features over the data axis
partitioned_dense = nn.Dense(features, kernel_init=with_partitioning(nn.initializers.lecun_normal, (None, "data")))
```

Initializing a model that creates partitioned weights would result in the following variable structure:

```python
variables = partitioned_dense.init(rng, jnp.ones((4,)))
jax.tree.map(np.shape, variables)  # => {"params": {"kernel": Partitioned(value=(4, 8), names=(None, "data")), bias: (8,)}}
```

The variable tree with metadata can be used to integrate with other libraries and APIs.
For example, we can turn the ``Partitioned`` metadata into ``jax.pjit`` sharding annotations:

```python
def to_sharding_spec(x):
  if isinstance(x, Partitioned):
    return PartitionSpec(*x.names)
  else:
    # fully replicated
    return PartitionSpec()

# Result: {"params": {"kernel": PartitionSpec(None, "data"), bias: PartitionSpec()}}
variables_pspec = jax.tree.map(to_sharding_spec, variables, is_leaf=lambda x: isinstance(x, Partitioned))
```

### Unbox syntax


Metadata typically doesn't need to be handled by Modules directly. Therefore, we propose to make Modules agnostic to Metadata boxes by default.
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

