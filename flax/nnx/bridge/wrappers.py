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

from functools import partial
import typing as tp
from typing import Any
import warnings
import dataclasses

from flax import linen
from flax import core
from flax import nnx
from flax.core import FrozenDict
from flax.core import meta
from flax.nnx import graph
from flax.nnx import variablelib
from flax.nnx.bridge import variables as bv
from flax.nnx.bridge import module as bdg_module
from flax.nnx.module import Module
from flax.nnx.statelib import State
from flax.nnx.pytreelib import Pytree
from flax.nnx.rnglib import Rngs
import jax
from jax import tree_util as jtu

M = tp.TypeVar('M', bound=Module)

# Flax-like style is NNX
@dataclasses.dataclass
class Functional(tp.Generic[M]):
  module_type: tp.Type[M]
  graphdef: tp.Optional[graph.GraphDef[M]]
  args: tuple[tp.Any, ...]
  kwargs: dict[str, tp.Any]

  def init(self, *, rngs: tp.Optional[Rngs] = None) -> State:
    kwargs = {}
    if rngs is not None:
      kwargs['rngs'] = rngs
    module = self.module_type(*self.args, **self.kwargs, **kwargs)
    graphdef, state = nnx.split(module)
    self.graphdef = graphdef
    return state  # type: ignore

  def apply(self, *states: tp.Any):
    assert self.graphdef is not None
    return self.graphdef.apply(*states)


def functional(cls: tp.Type[M]) -> tp.Callable[..., Functional[M]]:
  def _functional_constructor(*args: tp.Any, **kwargs: tp.Any) -> Functional[M]:
    return Functional(cls, None, args, kwargs)

  return _functional_constructor


def _set_initializing(module: Module, initializing: bool):
  for _, value in graph.iter_graph(module):
    if isinstance(value, Pytree):
      value._pytree__state._initializing = initializing


def lazy_init(fn: Module | tp.Callable[..., tp.Any], *args, **kwargs):
  """To run through an arbitrary nnx.Module method and initialize all its needed state.

  Here used to trigger initialization of all `LinenToNNX` module variables."""
  if isinstance(fn, Module):
    module = fn
    assert callable(fn)
  else:
    if not (hasattr(fn, '__self__') and isinstance(fn.__self__, Module)):
      raise ValueError(f'{fn = } needs to be a method of an NNX Module.')
    module = fn.__self__
  _set_initializing(module, True)
  try:
    _ = fn(*args, **kwargs)
  finally:
    _set_initializing(module, False)
  return fn

def current_linen_module() -> linen.Module | None:
  """Get the current Linen module from the Linen context."""
  if linen.module._context.module_stack:  # pylint: disable=W0212
    return linen.module._context.module_stack[-1]  # pylint: disable=W0212
  return None

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

    self.to_nnx__rngs: Rngs | None
    if isinstance(rngs, jax.Array):
      self.to_nnx__rngs = Rngs(params=rngs)
    elif isinstance(rngs, nnx.Rngs):
      self.to_nnx__rngs = rngs.fork()
    else:
      self.to_nnx__rngs = rngs

  @property
  def rngs(self) -> Rngs | None:
    warnings.warn(
        '`ToNNX.rngs` is deprecated. Please use `to_nnx__rngs` instead.',
        DeprecationWarning,
    )
    return self.to_nnx__rngs

  @property
  def module(self) -> linen.Module:
    warnings.warn(
        '`ToNNX.module` is deprecated. Please use `to_nnx__module` instead.',
        DeprecationWarning,
    )
    return self.to_nnx__module

  def lazy_init(self, *args, **kwargs):
    """A shortcut of calling `nnx.bridge.lazy_init()` upon this module."""
    return lazy_init(self, *args, **kwargs)

  def __getattr__(self, name: str):
    if hasattr(super(), name):
      return super().__getattribute__(name)
    maybe_method = getattr(type(self.to_nnx__module), name, None)
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
    mutable: tp.Any = None,
    **kwargs: Any,
  ) -> Any:
    # Shape-based lazy init of the flax variables
    if rngs is None:
      rngs = self.to_nnx__rngs
    if isinstance(rngs, nnx.Rngs):
      _rngs = {name: stream() for name, stream in rngs.items()}
    elif isinstance(rngs, jax.Array):
      _rngs = {'params': rngs}
    else:
      _rngs = {}
    # rename default to params
    if 'params' not in _rngs and 'default' in _rngs:
      _rngs['params'] = _rngs.pop('default')
    if self._pytree__state.initializing:
      out, updates = self.to_nnx__module.init_with_output(_rngs, *args, method=method, **kwargs)
    else:
      nnx_attrs = {
          k: v
          for k, v in vars(self).items()
          if not k.startswith('to_nnx__') and not k.startswith('_pytree__')
      }
      variables = bv.nnx_attrs_to_linen_vars(nnx_attrs)

      # Get `mutable` from top level bridge.Module context if any
      if mutable is not None:
        pass
      elif (m := bdg_module.current_module()) is not None:  # type: ignore[assignment]
        assert m.scope is not None
        mutable = m.scope.mutable
      elif (m := current_linen_module()) is not None:  # type: ignore[assignment]
        assert m.scope is not None
        mutable = m.scope.mutable
      else:
        mutable = False

      out = self.to_nnx__module.apply(
        variables, *args, rngs=_rngs, method=method, mutable=mutable, **kwargs
      )

      # Split out the updates if `mutable` is passed into the Flax module
      if mutable is not False:
        out, updates = out
      else:
        updates = None

    # Split out the updates if `mutable` is passed into the Flax module
    if updates:
      nnx_attrs = bv.linen_vars_to_nnx_attrs(updates)
      # nnx.update(self, nnx_attrs)
      # TODO(cgarciae): ideally we just do an update but currently dictionaries don't allow
      # insertion of new keys, we need to enable this in NNX to simplify the code bellow
      # to the simple nnx.update(self, nnx_attrs) above.
      for attr_name, value in nnx_attrs.items():
        if hasattr(self, attr_name) and isinstance(value, dict):
          original_value = getattr(self, attr_name)
          new_values = bv._recursive_merge(original_value, value)
          setattr(self, attr_name, nnx.data(new_values))
        else:
          setattr(self, attr_name, nnx.data(value))

    return out


def linen_rngs_dict(linen_module: linen.Module, add_default: bool = False):
  """Given a module, split out one of its every active RNG key collections."""
  assert linen_module.scope is not None, 'linen_rngs_dict() must be called inside a Linen module.'
  rngs: dict[str, tp.Any] = {
      name: linen_module.make_rng(name)
      for name in linen_module.scope.rngs.keys()
  }
  if add_default and 'default' not in rngs:
    rngs['default'] = 0
  return rngs

def _get_module_method(module, method: tp.Callable[..., Any] | str | None):
  """Get a callable method from the module, or raise TypeError."""
  if method is None:
    method = '__call__'

  if isinstance(method, str):
    attribute_name = method
    method = getattr(type(module), attribute_name)
    if not callable(method):
      class_name = type(module).__name__
      raise TypeError(
        f"'{class_name}.{attribute_name}' must be a callable, got"
        f' {type(method)}.'
      )
  if not callable(method):
    class_name = type(module).__name__
    raise TypeError(
      f"'{method}' must be a callable, got {type(method)}."
    )

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
  metadata_fn: tp.Callable[[variablelib.Variable], tp.Any] | None = bv.to_linen_var

  @linen.compact
  def __call__(
    self, *args, nnx_method: tp.Callable[..., Any] | str | None = None, **kwargs
  ):
    def _module_kwargs():
      maybe_add_default = not self.is_initializing()
      module_kwargs = dict(self.kwargs)
      if not self.skip_rng:
        module_kwargs['rngs'] = nnx.Rngs(
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
        list(core.unfreeze(self.variables).values()),  # type: ignore[wrong-arg-types, arg-type]
        is_leaf=lambda x: isinstance(x, meta.AxisMetadata),
    )
    if not states:
      states = ({},)

    new_state = nnx.merge_state(*states)
    new_state_flat = nnx.traversals.flatten_mapping(new_state)
    current_state_flat = nnx.traversals.flatten_mapping(nnx.state(module))
    unknown_state_flat = {path: v for path, v in new_state_flat.items() if path not in current_state_flat}

    if unknown_state_flat:
      paths_str = ""
      for path, _ in unknown_state_flat.items():
        paths_str += f"\n  - {'/'.join(map(str, path))}"

      warnings.warn(f"Found unknown module paths in incoming state:{paths_str}")

    nnx.update(module, new_state)

    method_fn = _get_module_method(module, nnx_method)
    out = method_fn(module, *args, **kwargs)
    self._update_variables(module)
    return out

  def __getattr__(self, name: str):
    if hasattr(super(), name):
      return super().__getattribute__(name)
    if name in self.kwargs:
      return self.kwargs[name]
    maybe_method = getattr(self.nnx_class, name, None)
    if callable(maybe_method):
      method = partial(self.__call__, nnx_method=maybe_method)
      method.__self__ = self
      return method
    return super().__getattribute__(name)

  def _update_variables(self, module):
    """Store the NNX module's graph def and state inside Linen module variables."""
    state = nnx.state(module, nnx.Not(nnx.RngState))

    collection_flat_state: dict[str, list[tuple[tuple[tp.Any, ...], tp.Any]]] = {}

    # group state by collection
    for path, leaf in nnx.to_flat_state(state):
      type_ = type(leaf)
      collection = variablelib.variable_name_from_type(
          type_, allow_register=True
      )
      if collection not in collection_flat_state:
        collection_flat_state[collection] = []
      collection_flat_state[collection].append((path, leaf))

    # update linen variables
    for collection, flat_state in collection_flat_state.items():
      if self.is_mutable_collection(collection):

        def _to_linen_var(x):
          if isinstance(x, nnx.Variable):
            if self.metadata_fn is not None:
              return self.metadata_fn(x)  # pylint: disable=too-many-function-args
            else:
              return x.get_value()
          return x

        collection_state = nnx.traversals.unflatten_mapping(flat_state)
        collection_state = jax.tree.map(
            _to_linen_var,
            collection_state,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )
        for k, v in collection_state.items():
          self.put_variable(collection, k, v)



class _Missing:
  ...


_MISSING = _Missing()


def to_linen(
    nnx_class: tp.Callable[..., Module],
    *args,
    metadata_fn: (
        tp.Callable[[variablelib.Variable], tp.Any] | None
    ) = bv.to_linen_var,
    name: str | None = None,
    skip_rng: bool = False,
    abstract_init: bool = True,
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

def to_linen_class(
    base_nnx_class: type[M],
    base_metadata_fn: tp.Callable[[variablelib.VariableState], tp.Any] | None = bv.to_linen_var,
    base_skip_rng: bool = False,
    **partial_kwargs: tp.Any,
) -> type[ToLinen]:
  """Dynamically wraps an NNX module class into a Flax Linen module class."""

  class ToLinenPartial(ToLinen):
    """A dynamically created Linen Module that wraps a specific NNX Module.

    This class is not meant to be used directly. Instead, it is created and
    returned by the `to_linen_class` function. It acts as a "partially applied"
    version of the `ToLinen` wrapper, where the NNX module to be wrapped and
    its default arguments are pre-configured.

    When you instantiate this class, it behaves like a standard Linen module.
    The arguments you provide during instantiation can override the defaults
    that were set when this class was created by `to_linen_class`.

    For example:
      >>> from flax import linen as nn, nnx
      >>> from maxtext.src.maxtext.layers import linears
      >>> # Create a specialized Linen wrapper for linears.DenseGeneral
      >>> LinenDenseGeneral = to_linen_class(linears.DenseGeneral)
      >>> # Now, LinenDenseGeneral can be used like a regular Linen module
      >>> class MyModel(nn.Module):
      ...   def setup(self):
      ...     # Instantiate the wrapped linears.DenseGeneral with its arguments
      ...     self.dense = LinenDenseGeneral(
      ...         in_features_shape=10, out_features_shape=5
      ...     )
      ...   def __call__(self, x):
      ...     return self.dense(x)

    Attributes:
      (The attributes are dynamically set by the `ToLinen` parent class based
       on the arguments provided during instantiation.)
    """

    def __init_subclass__(cls, **kwargs):
      super().__init_subclass__(**kwargs)

      def __init__(
          self,
          args=None,
          kwargs=None,
          nnx_class=None,
          skip_rng=None,
          metadata_fn=None,
          name=_MISSING,
          parent=_MISSING,
          **other_kwargs,
      ):
        linen_kwargs = {}
        if not isinstance(parent, _Missing):
          linen_kwargs["parent"] = parent
        if not isinstance(name, _Missing):
          linen_kwargs["name"] = name
        ToLinen.__init__(
            self,
            nnx_class=nnx_class or base_nnx_class,
            args=args or (),
            metadata_fn=metadata_fn or base_metadata_fn,
            skip_rng=skip_rng or base_skip_rng,
            kwargs=FrozenDict({**partial_kwargs, **(kwargs or {}), **other_kwargs}),
            **linen_kwargs,
        )

      cls.__init__ = __init__

  return ToLinenPartial
