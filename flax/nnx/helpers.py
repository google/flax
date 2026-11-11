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
from __future__ import annotations

import inspect
import typing as tp

import jax
import jax.numpy as jnp
import optax

from flax.nnx import graphlib, reprlib
from flax.nnx.graphlib import GraphDef
from flax.nnx.module import Module
from flax.nnx.proxy_caller import ApplyCaller
from flax.nnx.statelib import State
from flax.training.train_state import struct
from flax.nnx.pytreelib import data

A = tp.TypeVar('A')
M = tp.TypeVar('M', bound=Module)
TS = tp.TypeVar('TS', bound='TrainState')

class Dict(reprlib.MappingReprMixin, Module, tp.MutableMapping[str, A]):
  """A Module that implements a mutable mapping.

  This class provides a way to store and manipulate a mapping of keys to values
  contained a mixed set of data (e.g. Array, Variables, Modules) and static
  (e.g. functions, strings) types.

  Example:
    >>> from flax import nnx
    ...
    >>> rngs = nnx.Rngs(0)
    >>> layers = nnx.Dict({
    ...     'kernel1': nnx.Linear(1, 32, rngs=rngs),   # data
    ...     'activation1': nnx.relu,                   # static
    ...     'kernel2': nnx.Linear(32, 1, rngs=rngs),   # data
    ... })
  """
  @tp.overload
  def __init__(self, iterable: tp.Iterable[tp.Tuple[str, A]], /): ...

  @tp.overload
  def __init__(
    self, mapping: tp.Optional[tp.Mapping[str, A]] = None, /, **kwargs: A
  ): ...

  def __init__(self, *args, **kwargs):
    for name, value in dict(*args, **kwargs).items():
      setattr(self, name, value)

  def __getitem__(self, key) -> A:
    try:
      return getattr(self, key)
    except AttributeError as e:
      raise KeyError(key) from e

  def __setitem__(self, key, value):
    setattr(self, key, value)

  def __iter__(self) -> tp.Iterator[str]:
    return (k for k in vars(self) if not k.startswith('_pytree__'))

  def __len__(self) -> int:
    length = 0
    for _ in self:
      length += 1
    return length

  def __hash__(self) -> int:
    return id(self)

  def __delitem__(self, key: str) -> None:
    try:
      delattr(self, key)
    except AttributeError as e:
      raise KeyError(key) from e

  if tp.TYPE_CHECKING:
    def __getattr__(self, key: str) -> A:
      ...
    def __setattr__(self, key: str, value: A) -> None:
      ...

def List(a):
  return data(list(a))


class Sequential(Module, tp.Iterable[tp.Callable[..., tp.Any]]):
  """A Module that applies a sequence of callables.

  This class provides a way to store and manipulate a sequence of callables
  (e.g. layers, activation functions) and apply them in order. ``Sequential``
  is iterable over ``self.layers``, and keyword arguments passed to ``__call__``
  (including ``rngs``) are forwarded to every layer whose signature accepts
  them.

  Example::

    >>> from flax import nnx
    >>> import jax.numpy as jnp
    ...
    >>> rngs = nnx.Rngs(0)
    >>> model = nnx.Sequential(
    ...   nnx.Linear(1, 4, rngs=rngs),  # data
    ...   nnx.relu,                     # static
    ...   nnx.Linear(4, 2, rngs=rngs),  # data
    ... )
    >>> x = jnp.ones((1, 1))
    >>> y = model(x)
    >>> y.shape
    (1, 2)
  """

  def __init__(self, *fns: tp.Callable[..., tp.Any]):
    """
    Args:
      *fns: A sequence of callables to apply.
    """
    self.layers: List[tp.Callable[..., tp.Any]] = List(fns)

  def __iter__(self) -> tp.Iterator[tp.Callable[..., tp.Any]]:
    return self.layers.__iter__()

  def __call__(self, *args, **kwargs) -> tp.Any:
    if len(self.layers) == 0:
      if len(args) == 1:
        return args[0]
      elif len(args) > 0:
        return args
      elif len(kwargs) > 0:
        return kwargs
      else:
        return None

    output: tp.Any = None

    for i, f in enumerate(self.layers):
      if not callable(f):
        raise TypeError(f'Sequence[{i}] is not callable: {f}')
      if i > 0:
        if isinstance(output, tuple):
          args = output
          new_kwargs = None
        elif isinstance(output, dict):
          args = ()
          new_kwargs = dict(output)
        else:
          args = (output,)
          new_kwargs = None
      else:
        new_kwargs = None

      layer_kwargs = _filter_kwargs(f, len(args), kwargs) if kwargs else {}
      if new_kwargs is not None:
        if repeated_kwargs := set(new_kwargs).intersection(layer_kwargs):
          raise TypeError(
              f'Got multiple values for arguments {repeated_kwargs} in'
              f' Sequential[{i}]'
          )
        layer_kwargs = {**new_kwargs, **layer_kwargs}

      output = f(*args, **layer_kwargs)

    return output


class ModuleDefApply(tp.Protocol, tp.Generic[M]):
  def __call__(
    self, state: State, *states: State
  ) -> ApplyCaller[tuple[State, GraphDef[M]]]: ...


class TrainState(tp.Generic[M], struct.PyTreeNode):
  graphdef: graphlib.GraphDef[M]
  params: State
  opt_state: optax.OptState
  step: jax.Array
  tx: optax.GradientTransformation = struct.field(pytree_node=False)

  @classmethod
  def create(
    cls,
    graphdef: graphlib.GraphDef[M],
    *,
    params: State,
    tx: optax.GradientTransformation,
    step: int = 0,
    **kwargs,
  ):
    return cls(
      graphdef=graphdef,
      params=params,
      opt_state=tx.init(params),
      step=jnp.asarray(step),
      tx=tx,
      **kwargs,
    )

  if tp.TYPE_CHECKING:

    def __getattr__(self, key: str) -> tp.Any: ...

  def apply(
    self, state: tp.Union[State, str], *states: tp.Union[State, str]
  ) -> ApplyCaller[tuple[GraphDef[M], State]]:
    states = (state, *states)

    _states: list[State] = []

    for _state in states:
      if isinstance(_state, str):
        _state_key = _state
        _state = getattr(self, _state_key)
        if not isinstance(_state, State):
          raise TypeError(
            f'Expected {self.__class__.__name__}.{_state_key} to be a State, got {type(_state)}'
          )
      _states.append(_state)

    return self.graphdef.apply(*_states)

  def apply_gradients(self: TS, grads: State, **kwargs) -> TS:
    updates, opt_state = self.tx.update(grads, self.opt_state, self.params)
    params = optax.apply_updates(self.params, updates)  # type: ignore
    step = self.step + 1
    return self.replace(
      params=params,
      opt_state=opt_state,
      step=step,
      **kwargs,
    )


def _filter_kwargs(
    func: tp.Callable[..., tp.Any],
    num_pos_args: int,
    kwargs: dict[str, tp.Any],
) -> dict[str, tp.Any]:
  """Return the subset of ``kwargs`` accepted by ``func`` given ``num_pos_args``."""
  try:
    params = inspect.signature(func).parameters.values()
  except (ValueError, TypeError):
    return {}

  pos_index = 0
  bound_pos_or_kw: set[str] = set()
  accepted_kw: set[str] = set()
  has_var_kw = False

  for param in params:
    if param.kind == param.POSITIONAL_ONLY:
      pos_index += 1
    elif param.kind == param.POSITIONAL_OR_KEYWORD:
      if pos_index < num_pos_args:
        bound_pos_or_kw.add(param.name)
      else:
        accepted_kw.add(param.name)
      pos_index += 1
    elif param.kind == param.KEYWORD_ONLY:
      accepted_kw.add(param.name)
    elif param.kind == param.VAR_KEYWORD:
      has_var_kw = True

  if has_var_kw:
    return {k: v for k, v in kwargs.items() if k not in bound_pos_or_kw}
  return {k: v for k, v in kwargs.items() if k in accepted_kw}


def has_keyword_arg(func: tp.Callable[..., tp.Any], name: str) -> bool:
  """Return True if func has keyword-only arguments with the given name."""
  return any(
      param.name == name
      and param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
      for param in inspect.signature(func).parameters.values()
  )
