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

import typing as tp

from flax import struct
from flax.nnx.nnx import graph
from flax.nnx.nnx.proxy_caller import CallableProxy, DelayedAccessor

A = tp.TypeVar('A')
C = tp.TypeVar('C', covariant=True)


class PureCall(tp.Protocol, tp.Generic[C]):
  def __getattr__(self, __name) -> PureCall[C]: ...

  def __getitem__(self, __name) -> PureCall[C]: ...

  def __call__(self, *args, **kwargs) -> tuple[tp.Any, C]: ...


@struct.dataclass
class Pure(tp.Generic[A]):
  """A Pure pytree representation of a node.

  ``Pure`` objects are PyTrees that contain a ``GraphDef`` and
  a ``State`` which together represent a node's state. The Pure API
  converts mutable objects with implicit state updates into immutable
  objects with explicit state updates, making it easier to use with
  regular JAX code.

  Example::

    >>> from flax import nnx
    >>> import jax
    >>> import jax.numpy as jnp
    ...
    >>> class StatefulLinear(nnx.Module):
    ...   def __init__(self, din, dout, rngs):
    ...     self.w = nnx.Param(jax.random.uniform(rngs(), (din, dout)))
    ...     self.b = nnx.Param(jnp.zeros((dout,)))
    ...     self.count = nnx.Variable(jnp.array(0, dtype=jnp.uint32))
    ...
    ...   def increment(self):
    ...     self.count.value += 1
    ...
    ...   def __call__(self, x):
    ...     self.increment()
    ...     return x @ self.w + self.b
    ...
    >>> linear = StatefulLinear(3, 2, nnx.Rngs(0))
    >>> pure_linear = nnx.pure(linear)
    ...
    >>> @jax.jit
    ... def forward(x, pure_linear):
    ...   y, pure_linear = pure_linear(x)
    ...   return y, pure_linear
    ...
    >>> x = jnp.ones((1, 3))
    >>> y, pure_linear = forward(x, pure_linear)
    >>> y, pure_linear = forward(x, pure_linear)
    ...
    >>> linear = nnx.stateful(pure_linear)
    >>> linear.count.value
    Array(2, dtype=uint32)

  In this example the ``__call__`` was used but in general any method
  can be called. If the desired method is in a subnode, attribute access
  and indexing can be used to reach it::

    >>> rngs = nnx.Rngs(0)
    >>> nodes = dict(
    ...   a=StatefulLinear(3, 2, rngs),
    ...   b=StatefulLinear(2, 1, rngs),
    ... )
    ...
    >>> pure = nnx.pure(nodes)
    >>> _, pure = pure['b'].increment()
    >>> nodes = nnx.stateful(pure)
    ...
    >>> nodes['a'].count.value
    Array(0, dtype=uint32)
    >>> nodes['b'].count.value
    Array(1, dtype=uint32)
  """

  _pure_graphdef: graph.GraphDef[A]
  _pure_state: graph.GraphState

  def __call__(self, *args, **kwargs) -> tuple[tp.Any, Pure[A]]:
    node = graph.merge(self._pure_graphdef, self._pure_state)
    out = node(*args, **kwargs)  # type: ignore
    graphdef, state = graph.split(node)
    return out, Pure(graphdef, state)

  def __getattr__(self, name: str) -> PureCall[Pure[A]]:
    def pure_caller(accessor: DelayedAccessor, *args, **kwargs):
      node = graph.merge(self._pure_graphdef, self._pure_state)
      method = accessor(node)
      out = method(*args, **kwargs)
      graphdef, state = graph.split(node)
      return out, Pure(graphdef, state)

    proxy = CallableProxy(pure_caller)
    return getattr(proxy, name)

  def __getitem__(self, name: str) -> PureCall[Pure[A]]:
    def pure_caller(accessor: DelayedAccessor, *args, **kwargs):
      node = graph.merge(self._pure_graphdef, self._pure_state)
      method = accessor(node)
      out = method(*args, **kwargs)
      graphdef, state = graph.split(node)
      return out, Pure(graphdef, state)

    proxy = CallableProxy(pure_caller)
    return proxy[name]


def stateful(pure: Pure[A]) -> A:
  """Create a new node from a Pure object.

  Args:
    pure: The Pure object to create a node from.

  Returns:
    A new node with the same state as the Pure object.
  """
  return graph.merge(pure._pure_graphdef, pure._pure_state)


def pure(node: A) -> Pure[A]:
  """Creates a :class:`Pure` object from a node.

  Args:
    node: The node to create a Pure object from.

  Returns:
    A Pure object containing the node's state.
  """
  graphdef, state = graph.split(node)
  return Pure(graphdef, state)

# remove the replace method to avoid naming collisions in user code
del Pure.replace  # type: ignore