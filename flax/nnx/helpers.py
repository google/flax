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

from flax.nnx import graph
from flax.nnx.module import GraphDef, Module
from flax.nnx.proxy_caller import ApplyCaller
from flax.nnx.rnglib import Rngs
from flax.nnx.statelib import State
from flax.training.train_state import struct

A = tp.TypeVar('A')
M = tp.TypeVar('M', bound=Module)
TS = tp.TypeVar('TS', bound='TrainState')


class Dict(Module, tp.Mapping[str, A]):
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

  def __getattr__(self, key) -> A:
    return super().__getattribute__(key)

  def __setattr__(self, key, value):
    super().__setattr__(key, value)

  def __iter__(self) -> tp.Iterator[str]:
    return (k for k in vars(self) if k != '_object__state')

  def __len__(self) -> int:
    return len(vars(self))

  def __hash__(self) -> int:
    return id(self)


class Sequential(Module):
  
  @tp.overload
  def __init__(self, fns: tp.List[tp.Any]): ...
  
  def __init__(self, *fns: tp.Callable[..., tp.Any]):
    """A sequential container for modules.
    
    Args:
        fns (Callable[nnx.Module]): A list of Modules.
    """
    if len(fns) == 1 and isinstance(fns[0], (list, tuple)):
      fns = fns[0] 
    self.layers = list(fns)
  
  def __add__(self, other:'Sequential') -> 'Sequential':
    if isinstance(other, Sequential):
      result = Sequential()
      for module in self.layers:
          result.layers.append(module)
      for module in other.layers:
          result.layers.append(module)
      return result
    else:
      raise ValueError(
        "The add operator only supports objects of the Sequential class, "
        f"but {type(other)} is given."
      )
      
  def __iadd__(self, other:'Sequential') -> tp.Self:
    if isinstance(other, Sequential):
      for module in other.layers:
        self.layers.append(module)
      return self
    else:
      raise ValueError(
        "The add operator only supports objects of the Sequential class, "
        f"but {type(other)} is given."
      )
  
  def __mul__(self, other:int) -> 'Sequential':
    if not isinstance(other, int):
      raise ValueError(
        "The multiply operator only supports integers, "
        f"but {type(other)} is given."
      )
      
    elif other < 0:
      raise ValueError(
        "The multiply operator only supports positive integers, "
        f"but {other} is given."
      )
      
    else:
      result = Sequential()
      for _ in range(other):
        for module in self.layers:
          result.layers.append(module)
      return result
  
  def __rmul__(self, other:int) -> 'Sequential':
    return self.__mul__(other)
  
  def __imul__(self, other: int) -> tp.Self:
    if not isinstance(other, int):
        raise ValueError(
            "The multiply operator only supports integers, "
            f"but {type(other)} is given."
        )
    elif other < 0:
        raise ValueError(
            "The multiply operator only supports positive integers, "
            f"but {other} is given."
        )
    else:
        original = self.layers.copy()
        self.layers = original * other
        del original
        return self
      
  def __iter__(self) -> tp.Iterable[Module]:
    return iter(self.layers)
  
  def __getitem__(self, indx: tp.Union[int, slice]) -> tp.Union[Module, 'Sequential']:
    if isinstance(indx, slice):
      return Sequential(*self.layers[indx])
    return self.layers[indx]
    
  def __setitem__(self, indx:int, module:Module) -> None:
    
    if indx > len(self.layers):
      raise IndexError("Index out of range")
    
    if not isinstance(module, Module):
      raise TypeError(f"Expected a Module, but got {type(module)}")

    self.layers[indx] = module
    
  def __delitem__(self, indx:int) -> None:
    if indx > len(self.layers):
      raise IndexError("Index out of range")
    del self.layers[indx]
  
  def __len__(self) -> int:
    return len(self.layers)
  
  def pop(self, indx:int) -> Module:
    """Remove and return the module at the given index in the Sequential."""
    return self.layers.pop(indx)
  
  def append(self, module:Module) -> 'Sequential':
    """Append a module to the end of the Sequential."""
    if not isinstance(module, Module):
      raise TypeError(f"Expected a Module, but got {type(module)}")
    return self.layers.append(module)
  
  def extend(self, modules:tp.List[Module] | 'Sequential') -> 'Sequential':
    """Extend the Sequential by appending modules from the list or Sequential."""
    if not isinstance(modules, (list, Sequential)):
      raise TypeError(f"Expected a list or Sequential, but got {type(modules)}")
    self.layers.extend(modules)
    return self
    
  def insert(self, indx: int, module: Module) -> None:
    """Insert a module at the given index in the Sequential."""
    if not isinstance(module, Module):
        raise TypeError(f"Expected a Module, but got {type(module)}")
    
    n = len(self.layers)
    if not (-n <= indx <= n):
        raise IndexError("Index out of range")
        
    self.layers.insert(indx, module)
    return self

  def __call__(self, *args, rngs: tp.Optional[Rngs] = None, **kwargs) -> tp.Any:
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
