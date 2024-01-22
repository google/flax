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

# Copyright 2023 The Flax Authors.
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
import numpy as np
import optax

from flax.experimental.nnx.nnx import pytreelib
from flax.experimental.nnx.nnx.module import GraphDef, Module
from flax.experimental.nnx.nnx.proxy_caller import ApplyCaller
from flax.experimental.nnx.nnx.rnglib import Rngs
from flax.experimental.nnx.nnx.state import State

A = tp.TypeVar('A')
M = tp.TypeVar('M', bound=Module)


class Dict(Module, tp.Mapping[str, A]):
  @tp.overload
  def __init__(self, __iterable: tp.Iterable[tp.Tuple[str, A]]):
    ...

  @tp.overload
  def __init__(
    self, __mapping: tp.Optional[tp.Mapping[str, A]] = None, **kwargs: A
  ):
    ...

  def __init__(self, *args, **kwargs):
    for name, value in dict(*args, **kwargs).items():
      setattr(self, name, value)

  def __getitem__(self, key) -> A:
    return getattr(self, key)

  def __setitem__(self, key, value):
    setattr(self, key, value)

  def __getattr__(self, key) -> A:
    return super().__getattribute__(key)

  def __setattr__(self, key, value):
    super().__setattr__(key, value)

  def __iter__(self) -> tp.Iterator[str]:
    return (k for k in vars(self) if k != '_module__state')

  def __len__(self) -> int:
    return len(vars(self))


class Sequence(Module, tp.Generic[A]):
  def __init__(self, layers: tp.Iterable[A]):
    i = 0
    for i, value in enumerate(layers):
      setattr(self, str(i), value)
    self._length = i + 1

  def __getitem__(self, key: int) -> A:
    if key >= len(self):
      raise IndexError(f'index {key} out of range for {self}')
    return getattr(self, str(key))

  def __setitem__(self, key: int, value: A):
    if key >= len(self):
      raise IndexError(f'index {key} out of range for {self}')
    setattr(self, str(key), value)

  def __iter__(self) -> tp.Iterator[A]:
    for i in range(len(self)):
      yield getattr(self, str(i))

  def __len__(self) -> int:
    return self._length

  def __call__(self, *args, rngs: tp.Optional[Rngs] = None, **kwargs) -> tp.Any:
    output: tp.Any = None

    for i, f in enumerate(self):
      if not callable(f):
        raise TypeError(f'Sequence[{i}] is not callable: {f}')
      if i > 0:
        if isinstance(output, tp.Tuple):
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
  ) -> ApplyCaller[tuple[State, GraphDef[M]]]:
    ...


class TrainState(pytreelib.Pytree, tp.Generic[M]):
  def __init__(
    self,
    graphdef: GraphDef[M],
    *,
    params: State,
    tx: optax.GradientTransformation,
    step: int = 0,
    **kwargs,
  ):
    self.graphdef = graphdef
    self.params: State = pytreelib.TreeNode(params)
    self.tx = tx
    self.opt_state = pytreelib.TreeNode(tx.init(self.params))
    self.step = pytreelib.TreeNode(jnp.asarray(step))
    for name, value in kwargs.items():
      if isinstance(value, (jax.Array, np.ndarray, State)):
        value = pytreelib.TreeNode(value)
      setattr(self, name, value)

  if tp.TYPE_CHECKING:

    def __getattr__(self, key: str) -> tp.Any:
      ...

  def apply(
    self, state: tp.Union[State, str], *states: tp.Union[State, str]
  ) -> ApplyCaller[tuple[State, GraphDef[M]]]:
    states = (state, *states)

    _states = (
      getattr(self, state) if isinstance(state, str) else state
      for state in states
    )

    return self.graphdef.apply(*_states)

  def apply_gradients(self, grads: State, **kwargs) -> 'TrainState[M]':
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
