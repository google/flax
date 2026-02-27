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

from flax.nnx import graph, reprlib
from flax.nnx.graph import GraphDef
from flax.nnx.module import Module
from flax.nnx.proxy_caller import ApplyCaller
from flax.nnx.rnglib import Rngs
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
    return getattr(self, key)

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
    delattr(self, key)

  if tp.TYPE_CHECKING:
    def __getattr__(self, key: str) -> A:
      ...
    def __setattr__(self, key: str, value: A) -> None:
      ...

def List(a):
  return data(list(a))


class Sequential(Module):
  """A Module that applies a sequence of callables.

  This class provides a way to store and manipulate a sequence of callables
  (e.g. layers, activation functions) and apply them in order.

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
    self.layers = List(fns)

  def __call__(self, *args, rngs: tp.Optional[Rngs] = None, **kwargs) -> tp.Any:
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
          kwargs = {}
        elif isinstance(output, dict):
          args = ()
          kwargs = output
        else:
          args = (output,)
          kwargs = {}
      if rngs is not None and has_keyword_arg(f, 'rngs'):
        kwargs['rngs'] = rngs

      output = f(*args, **kwargs)

    return output


class ModuleDefApply(tp.Protocol, tp.Generic[M]):
  def __call__(
    self, state: State, *states: State
  ) -> ApplyCaller[tuple[State, GraphDef[M]]]: ...


class TrainState(tp.Generic[M], struct.PyTreeNode):
  graphdef: graph.GraphDef[M]
  params: State
  opt_state: optax.OptState
  step: jax.Array
  tx: optax.GradientTransformation = struct.field(pytree_node=False)

  @classmethod
  def create(
    cls,
    graphdef: graph.GraphDef[M],
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


def has_keyword_arg(func: tp.Callable[..., tp.Any], name: str) -> bool:
  """Return True if func has keyword-only arguments with the given name."""
  return any(
    param.name == name
    and param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
    for param in inspect.signature(func).parameters.values()
  )
