# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NNX <> Linen interoperability."""

from functools import partial
import typing as tp
from typing import Any

from flax import linen
from flax import nnx
from flax.core import FrozenDict
from flax.core import meta
from flax.nnx import graph
from flax.nnx import variablelib
from flax.nnx.bridge import module as bdg_module
from flax.nnx.module import Module
from flax.nnx.object import Object
from flax.nnx.rnglib import Rngs
import jax
from jax import tree_util as jtu

M = tp.TypeVar("M", bound=Module)


def is_vanilla_variable(vs: variablelib.VariableState) -> bool:
  """A variables state is vanilla if its metadata is essentially blank.

  Returns False only if it has non-empty hooks or any non-built-in attribute.
  """
  for key, value in vs.get_metadata().items():
    if key.endswith("_hooks"):
      if value != ():
        return False
    else:
      return False
  return True


def to_linen_var(vs: variablelib.VariableState) -> meta.AxisMetadata:
  metadata = vs.get_metadata()
  if "linen_meta_type" in metadata:
    linen_type = metadata["linen_meta_type"]
    if hasattr(linen_type, "from_nnx_metadata"):
      return linen_type.from_nnx_metadata({"value": vs.value, **metadata})
    return linen_type(vs.value, **metadata)
  if is_vanilla_variable(vs):
    return vs.value
  return nnx.bridge.NNXMeta(vs.type, vs.value, metadata)


def get_col_name(keypath: tp.Sequence[Any]) -> str:
  """Given the keypath of a Flax variable type, return its Linen collection name."""
  # Infer variable type from the leaf's path, which contains its Linen collection name
  assert isinstance(keypath[0], jax.tree_util.DictKey)
  return str(keypath[0].key)


def to_nnx_var(col: str, x: meta.AxisMetadata | Any) -> variablelib.Variable:
  """Convert a Linen variable to an NNX variable."""
  vtype = variablelib.variable_type_from_name(col, allow_register=True)
  if isinstance(x, nnx.bridge.NNXMeta):
    assert vtype == x.var_type, (
      f"Type stored in NNXMeta {x.var_type} != type inferred from collection name {vtype}"
    )
    return x.to_nnx_variable()
  if isinstance(x, meta.AxisMetadata):
    x_metadata = vars(x)
    if hasattr(x, "to_nnx_metadata"):
      x_metadata = x.to_nnx_metadata()
    assert hasattr(x, "value")
    return vtype(**x_metadata, linen_meta_type=type(x))
  return vtype(x)


def _recursive_merge(dict1, dict2):
  """Recursively merge two dicts."""
  flat_map = nnx.traversals.flatten_mapping(dict1)
  flat_map |= nnx.traversals.flatten_mapping(dict2)
  return nnx.traversals.unflatten_mapping(flat_map)


def linen_vars_to_nnx_attrs(variables: tp.Mapping[str, Any]) -> dict[str, Any]:
  """Convert a dict of Linen-style variables to NNX variables."""
  nnx_vars = jax.tree_util.tree_map_with_path(
    lambda kp, x: to_nnx_var(get_col_name(kp), x),
    variables,
    is_leaf=lambda x: not isinstance(x, dict),
  )

  flat_paths: dict[tuple, tp.Any] = {}

  for col_name, col_variables in nnx_vars.items(): # pylint: disable=unused-variable
    for path, variable in nnx.traversals.flatten_mapping(col_variables).items():
      if path in flat_paths:
        raise ValueError(
          f"Found duplicate variable path {path} with variables "
          f"{flat_paths[path]} and {variable}. "
          "This is not allowed in NNX."
        )
      flat_paths[path] = variable

  nnx_vars = nnx.traversals.unflatten_mapping(flat_paths)
  return nnx_vars


def nnx_attrs_to_linen_vars(nnx_attrs: dict) -> dict:
  """Convert a dict of NNX variables (or variable states) to Linen-style variables."""
  linen_structured = {}
  for kp, v in nnx.traversals.flatten_mapping(nnx_attrs).items():
    if isinstance(v, variablelib.Variable):
      col_name = variablelib.variable_name_from_type(type(v))
      v = to_linen_var(v.to_state())
    elif isinstance(v, variablelib.VariableState):
      col_name = variablelib.variable_name_from_type(v.type)
      v = to_linen_var(v)
    else:
      raise ValueError(f"Cannot infer collection name from value: {v}")
    linen_structured[(col_name, *kp)] = v
  variables = nnx.traversals.unflatten_mapping(linen_structured)
  return variables


def _set_initializing(module: Module, initializing: bool):
  for _, value in graph.iter_graph(module):
    if isinstance(value, Object):
      value._object__state._initializing = initializing # pylint: disable=protected-access


def lazy_init(fn: Module | tp.Callable[..., tp.Any], *args, **kwargs):
  """To run through an arbitrary nnx.Module method and initialize all its needed state.

  Here used to trigger initialization of all `LinenToNNX` module variables."""
  if isinstance(fn, Module):
    module = fn
    assert callable(fn)
  else:
    if not (hasattr(fn, "__self__") and isinstance(fn.__self__, Module)):
      raise ValueError(f"{fn = } needs to be a method of an NNX Module.")
    module = fn.__self__
  _set_initializing(module, True)
  try:
    _ = fn(*args, **kwargs)
  finally:
    _set_initializing(module, False)
  return fn


class ToNNX(Module):
  """A wrapper to turn any Linen module into an NNX module.

  The result NNX module can be used standalone with all NNX APIs, or as a submodule of
  another NNX module.

  Since Linen module initialization requires a sample input, you need to call `lazy_init`
  with an argument to initialize the variables.

  Example::

    >>> from flax import linen as nn, nnx
    >>> import jax
    >>> linen_module = nn.Dense(features=64)
    >>> x = jax.numpy.ones((1, 32))
    >>> # Like Linen init(), initialize with a sample input
    >>> model = nnx.bridge.ToNNX(linen_module, rngs=nnx.Rngs(0)).lazy_init(x)
    >>> # Like Linen apply(), but using NNX's direct call method
    >>> y = model(x)
    >>> model.kernel.shape
    (32, 64)

  Args:
    module: The Linen Module instance.
    rngs: The `nnx.Rngs` instance being passed to any NNX module.

  Returns:
    A stateful NNX module that behaves the same as the wrapped Linen module.
  """

  def __init__(
    self,
    module: linen.Module,
    rngs: Rngs | jax.Array | None = None,
  ):
    self.to_nnx__module = module

    if isinstance(rngs, jax.Array):
      self.to_nnx__rngs = Rngs(params=rngs)
    elif isinstance(rngs, nnx.Rngs):
      self.to_nnx__rngs = rngs.fork() if hasattr(rngs, "fork") else nnx.clone(rngs)
    else:
      self.to_nnx__rngs = rngs

  def lazy_init(self, *args, **kwargs):
    """A shortcut of calling `nnx.bridge.lazy_init()` upon this module."""
    return lazy_init(self, *args, **kwargs)

  def __getattr__(self, name: str):
    if hasattr(super(), name):
      return super().__getattribute__(name)
    maybe_method = getattr(self.to_nnx__module.__class__, name, None)
    if callable(maybe_method):
      method = partial(self.__call__, method=maybe_method)
      method.__self__ = self
      return method
    return super().__getattribute__(name)

  def __call__(
    self,
    *args: Any,
    rngs: Rngs | jax.Array | None = None,
    method: tp.Callable[..., Any] | str | None = None,
    **kwargs: Any,
  ) -> Any:
    # Shape-based lazy init of the flax variables
    if rngs is None:
      rngs = self.to_nnx__rngs
    if isinstance(rngs, nnx.Rngs):
      _rngs = {name: stream() for name, stream in rngs.items()}
    elif isinstance(rngs, jax.Array):
      _rngs = {"params": rngs}
    else:
      _rngs = {}
    # rename default to params
    if "params" not in _rngs and "default" in _rngs:
      _rngs["params"] = _rngs.pop("default")
    if self._object__state.initializing:
      out, updates = self.to_nnx__module.init_with_output(
        _rngs, *args, method=method, **kwargs
      )
    else:
      nnx_attrs = {
        k: v
        for k, v in vars(self).items()
        if not k.startswith("to_nnx__") and not k.startswith("_object__")
      }
      variables = nnx_attrs_to_linen_vars(nnx_attrs)

      # Get `mutable` from top level bridge.Module context if any
      if (m := bdg_module.current_module()) is not None:
        assert m.scope is not None
        mutable = m.scope.mutable
        if "mutable" in kwargs and kwargs["mutable"] != mutable:
          raise ValueError(
            f"Multiple `mutable` arguments detected: {mutable} at top level vs "
            f"{kwargs['mutable']} in ToNNX() call"
          )
        kwargs["mutable"] = mutable

      out = self.to_nnx__module.apply(
        variables, *args, rngs=_rngs, method=method, **kwargs
      )

      # Split out the updates if `mutable` is passed into the Flax module
      if kwargs.get("mutable", False) is not False:
        out, updates = out
      else:
        updates = None

    # Split out the updates if `mutable` is passed into the Flax module
    if updates:
      nnx_attrs = linen_vars_to_nnx_attrs(updates)
      for attr_name, value in nnx_attrs.items():
        if hasattr(self, attr_name) and isinstance(value, dict):
          original_value = getattr(self, attr_name)
          new_values = _recursive_merge(original_value, value)
          setattr(self, attr_name, new_values)
        else:
          setattr(self, attr_name, value)

    return out


def linen_rngs_dict(linen_module: linen.Module, add_default: bool = False):
  """Given a module, split out one of its every active RNG key collections."""
  assert linen_module.scope is not None, (
    "linen_rngs_dict() must be called inside a Linen module."
  )
  rngs: dict[str, tp.Any] = {
    name: linen_module.make_rng(name) for name in linen_module.scope.rngs.keys()
  }
  if add_default and "default" not in rngs:
    rngs["default"] = 0
  return rngs


def _get_module_method(module, method: tp.Callable[..., Any] | str | None):
  """Get a callable method from the module, or raise TypeError."""
  if method is None:
    method = "__call__"

  if isinstance(method, str):
    attribute_name = method
    method = getattr(type(module), attribute_name)
    if not callable(method):
      class_name = type(module).__name__
      raise TypeError(
        f"'{class_name}.{attribute_name}' must be a callable, got {type(method)}."
      )
  if not callable(method):
    class_name = type(module).__name__
    raise TypeError(f"'{method}' must be a callable, got {type(method)}.")

  return method


class ToLinen(linen.Module):
  """A wrapper to turn any NNX module into a Linen module.

  The result Linen module can be used standalone with all Linen APIs, or as a
  submodule of
  another Linen module.

  Since NNX modules are stateful and owns the state, we only create it once
  during init
  time, and will track its state and static data as separate variables.

  Example::

    >>> from flax import linen as nn, nnx
    >>> import jax
    >>> model = nnx.bridge.ToLinen(nnx.Linear, args=(32, 64))
    >>> x = jax.numpy.ones((1, 32))
    >>> y, variables = model.init_with_output(jax.random.key(0), x)
    >>> y.shape
    (1, 64)
    >>> variables['params']['kernel'].shape
    (32, 64)
    >>> # The static GraphDef of the underlying NNX module
    >>> variables.keys()
    dict_keys(['params'])

  Args:
    nnx_class: The NNX Module class (not instance!).
    args: The arguments that normally would be passed in to create the NNX
      module.
    kwargs: The keyword arguments that normally would be passed in to create the
      NNX module.
    skip_rng: True if this NNX module doesn't need `rngs` arg during
      initialization (not common).

  Returns:
    A stateful NNX module that behaves the same as the wrapped Linen module.
  """

  nnx_class: tp.Callable[..., Module]
  args: tp.Sequence = ()
  kwargs: tp.Mapping[str, tp.Any] = FrozenDict({})
  skip_rng: bool = False
  metadata_fn: tp.Callable[[variablelib.VariableState], tp.Any] | None = to_linen_var

  @linen.compact
  def __call__(
    self, *args, nnx_method: tp.Callable[..., Any] | str | None = None, **kwargs
  ):
    module_kwargs = dict(self.kwargs)
    maybe_add_default = not self.is_initializing()

    def _module_kwargs():
      if not self.skip_rng:
        module_kwargs["rngs"] = nnx.Rngs(
          **linen_rngs_dict(self, add_default=maybe_add_default)
        )
      return module_kwargs

    # init codepath
    if self.is_initializing():
      module = self.nnx_class(*self.args, **_module_kwargs())
      # TODO: add lazy_init here in case there's an `ToNNX` submodule under `module`.
      # update linen variables before call module to save initial state
      self._update_variables(module)
      method_fn = _get_module_method(module, nnx_method)
      out = method_fn(module, *args, **kwargs)
      return out

    # create the nnx module
    module = self.nnx_class(*self.args, **_module_kwargs())
    # update nnx module from linen variables
    def maybe_unbox(x):
      if isinstance(x, meta.AxisMetadata):
        return x.unbox()
      return x
    states = jtu.tree_map(
      maybe_unbox,
      list(self.variables.values()),
      is_leaf=lambda x: isinstance(x, meta.AxisMetadata),
    )
    if not states:
      states = ({},)
    nnx.update(module, *states)

    method_fn = _get_module_method(module, nnx_method)
    out = method_fn(module, *args, **kwargs)
    self._update_variables(module)
    return out

  def __getattr__(self, name: str):
    if hasattr(super(), name):
      return super().__getattribute__(name)
    maybe_method = getattr(self.nnx_class, name, None)
    if callable(maybe_method):
      method = partial(self.__call__, nnx_method=maybe_method)
      method.__self__ = self
      return method
    return super().__getattribute__(name)

  def _update_variables(self, module):
    """Store the NNX module's graph def and state inside Linen module variables."""
    state = nnx.state(module, nnx.Not(nnx.RngState))

    collection_flat_state: dict[str, list[tuple[tuple[str, ...], tp.Any]]] = {}

    # group state by collection
    for path, leaf in nnx.to_flat_state(state):
      type_ = leaf.type if isinstance(leaf, nnx.VariableState) else type(leaf)
      collection = variablelib.variable_name_from_type(type_, allow_register=True)
      if collection not in collection_flat_state:
        collection_flat_state[collection] = []
      collection_flat_state[collection].append((path, leaf))

    # update linen variables
    for collection, flat_state in collection_flat_state.items():
      if self.is_mutable_collection(collection):

        def _to_linen_var(x):
          if isinstance(x, nnx.VariableState):
            if self.metadata_fn is not None:
              return self.metadata_fn(x) # pylint: disable=too-many-function-args
            else:
              return x.value
          return x

        collection_state = nnx.traversals.unflatten_mapping(flat_state)
        collection_state = jax.tree.map(
          _to_linen_var,
          collection_state,
          is_leaf=lambda x: isinstance(x, nnx.VariableState),
        )
        for k, v in collection_state.items():
          self.put_variable(collection, k, v)


def to_linen(
  nnx_class: tp.Callable[..., Module],
  *args,
  metadata_fn: (
    tp.Callable[[variablelib.VariableState], tp.Any] | None
  ) = to_linen_var,
  name: str | None = None,
  skip_rng: bool = False,
  **kwargs,
):
  """Shortcut of `nnx.bridge.ToLinen` if user is not changing any of its default fields."""
  return ToLinen(
    nnx_class,
    args=args,
    kwargs=FrozenDict(kwargs),
    metadata_fn=metadata_fn,
    skip_rng=skip_rng,
    name=name,
  )
