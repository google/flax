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

from flax.nnx.module import GraphDef, Module
from flax.nnx.proxy_caller import ApplyCaller
from flax.nnx.rnglib import Rngs
from flax.nnx.statelib import State
from flax.training.train_state import struct

A = tp.TypeVar('A')
M = tp.TypeVar('M', bound=Module)
TS = tp.TypeVar('TS', bound='TrainState')


class Dict(Module, tp.Mapping[str, A]):
  """ A simple module that behaves like a dictionary.

    This class provides a basic implementation of a dictionary-like data structure,
    allowing for the storage and retrieval of key-value pairs. It supports common
    dictionary operations such as item access, assignment, deletion, and iteration
    over keys, values, and items.

    Attributes:
        data (dict): The internal storage for the dictionary.
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

  def __getattr__(self, key) -> A:
    return super().__getattribute__(key)

  def __setattr__(self, key, value):
    super().__setattr__(key, value)

  def __iter__(self) -> tp.Iterator[str]:
    return (k for k in vars(self) if k != '_object__state')

  def __len__(self) -> int:
    return len(vars(self))

class Sequential(Module):
  """A module that applies a sequence of functions in order.

    This class provides a basic implementation of a sequential module, which
    applies a sequence of functions in order to an input. The functions can be
    any callable object, such as a function, method, or module. The output of
    each function is passed as input to the next function in the sequence.

    Attributes:
        layers (list): The sequence of functions to apply.
    """
  def __init__(self, *fns: tp.Callable[..., tp.Any]):
    self.layers = list(fns)

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
  """A dataclass that holds the state of a training loop.

    This class provides a basic implementation of a training state, which holds
    the state of a training loop. It contains the parameters of a model, the
    optimizer state, the current step, and the gradient transformation function
    used to update the parameters.

    Attributes:
        graphdef (GraphDef): The graph definition of the model.
        params (State): The parameters of the model.
        opt_state (optax.OptState): The optimizer state.
        step (int): The current step of the training loop.
        tx (optax.GradientTransformation): The gradient transformation function.
    """
  graphdef: GraphDef[M]
  params: State
  opt_state: optax.OptState
  step: jax.Array
  tx: optax.GradientTransformation = struct.field(pytree_node=False)

  @classmethod
  def create(
    cls,
    graphdef: GraphDef[M],
    *,
    params: State,
    tx: optax.GradientTransformation,
    step: int = 0,
    **kwargs,
  ):
    """Creates a new training state.

    This function creates a new training state with the given graph definition,
    model parameters, optimizer state, and gradient transformation function. It
    is typically used to initialize the state of a training loop before training
    a model.

    Args:
        graphdef: The graph definition of the model.
        params: The parameters of the model.
        tx: The gradient transformation function used to update the model parameters.
        step: The current step of the training loop.
        kwargs: Additional keyword arguments to pass to the training state.
    
    Returns:
        A new training state with the given parameters.
    """
    
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
    """Applies the model to a set of states.

    This function applies the model to a set of states, producing a new state
    that contains the output of the model. It is typically used to compute the
    output of a model given a set of input states.

    Args:
        state: The state to apply the model to.
        states: Additional states to apply the model to.
    
    Returns:
        A new state containing the output of the model.
    """
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
    """
    Applies gradients to the optimizer to update model parameters.

    This function takes an optimizer and a set of gradients, and applies the gradients
    to the optimizer to update the model parameters accordingly. It is typically used
    during the training loop to adjust the model's weights based on the computed gradients.

    Args:
        optimizer: The optimizer instance used to update the model parameters.
        gradients: A list or dictionary of gradients to be applied to the optimizer.

    Returns:
        The updated optimizer with the applied gradients.
    """
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
  """Return True if func has keyword-only arguments with the given name.

    This function checks if a callable object has any keyword-only arguments with
    the given name. It is used to determine if a function accepts a specific keyword
    argument, which is useful for handling optional arguments in a generic way.

    Args:
        func: The callable object to check for keyword-only arguments.
        name: The name of the keyword argument to check for.

    Returns:
        True if func has a keyword-only argument with the given name, False otherwise.
    """
  return any(
    param.name == name
    and param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
    for param in inspect.signature(func).parameters.values()
  )
