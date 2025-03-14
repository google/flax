# Copyright 2025 The Flax Authors.
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

import jax
import treescope  # type: ignore[import-untyped]

from flax.nnx import graph, proxy_caller, statelib
import typing as tp

from flax.nnx import reprlib

A = tp.TypeVar('A')


class Missing:
  pass

MISSING = Missing()

class Pytree(tp.Generic[A], reprlib.Representable):
  """
  A pytree-representable of an NNX object that can be used for
  convenience when interacting with JAX APIs that expect
  stateless callable pytree objects.

  Example::

    >>> from flax import nnx
    >>> from jax.experimental import ode
    ...
    >>> model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    >>> jax_model = nnx.Pytree(model)
    ...
    >>> y0 = jax.numpy.ones((2,))
    >>> def dy_dt(y, t, jax_model):
    ...   return model(y)
    >>> t = jax.numpy.linspace(0, 1, 10)
    >>> y = ode.odeint(dy_dt, y0, t, jax_model)
    >>> print(y.shape)
    (10, 2)

  Example::

    >>> from flax import nnx
    ...
    >>> class Model(nnx.Module):
    ...
    ...   def __init__(self, rngs: nnx.Rngs):
    ...     self.linear = nnx.Linear(2, 16, rngs=rngs)
    ...     self.dropout = nnx.Dropout(0.2, rngs=rngs)
    ...
    ...   def __call__(self, x):
    ...     return nnx.relu(self.dropout(self.linear(x)))
    ...
    >>> nnx_model = Model(nnx.Rngs(0))
    >>> jax_model = nnx.Pytree(nnx_model)
    ...
    >>> @jax.jit
    ... def f(jax_model, x):
    ...   return jax_model(x)
    ...
    >>> x = jax.numpy.ones((1, 2))
    >>> y = f(jax_model, x)
    >>> print(y.shape)
    (1, 16)

  Pytree objects frozen pytree-views of the original object, this means
  that they will always produce the exact output for a give input, any
  state updates performed during the call will be discarded::

    >>> import jax.numpy as jnp
    ...
    >>> y1 = f(jax_model, x)
    >>> y2 = f(jax_model, x)
    >>> assert jnp.allclose(y1, y2)

  If you want to keep the state updates, you can use the `merge` method
  and return a new ``Pytree`` object::

    >>> @jax.jit
    ... def g(jax_model, x):
    ...   nnx_model = jax_model.merge()
    ...   y = nnx_model(x)
    ...   return y, nnx.Pytree(nnx_model)
    ...
    >>> y1, jax_model = g(jax_model, x)
    >>> y2, jax_model = g(jax_model, x)
    >>> assert not jnp.allclose(y1, y2)

  Apart from using the ``__call__`` method, all methods from the
  original object or even methods from nested objects can also be
  invoked by simply using attribute access::

    >>> class MyModule(nnx.Module):
    ...   def __init__(self, rngs: nnx.Rngs):
    ...     self.encoder = nnx.Linear(2, 3, rngs=rngs)
    ...     self.decoder = nnx.Linear(3, 2, rngs=rngs)
    ...
    ...   def encode(self, x):
    ...     return self.encoder(x)
    ...
    ...   def decode(self, x):
    ...     return self.decoder(x)
    ...
    >>> nnx_model = MyModule(nnx.Rngs(0))
    >>> jax_model = nnx.Pytree(nnx_model)
    ...
    >>> @jax.jit
    ... def encode(jax_model, x):
    ...   return jax_model.encode(x) # or jax_model.encoder(x)
    ...
    >>> @jax.jit
    ... def decode(jax_model, x):
    ...   return jax_model.decode(x) # or jax_model.decoder(x)
    ...
    >>> x = jax.numpy.ones((1, 2))
    >>> z = encode(jax_model, x)
    >>> print(z.shape)
    (1, 3)
    >>> y = decode(jax_model, z)
    >>> print(y.shape)
    (1, 2)

  Internally, the object is store as `graphdef` and a `state`, these
  can be accessed using the `graphdef` and `state` properties::

    >>> graphdef, state = jax_model.graphdef, jax_model.state

  The representation of the ``state`` can be controlled by using the
  ``flatten`` and ``keep_metadata`` arguments. By default, the state is
  flattened and metadata is discarded, which yields maximum performance
  when using the object in a pure JAX context. Here's a table of the
  different configurations for ``state``::

  * ``flatten=True, keep_metadata=False``: ``list[Array]`` (default)
  * ``flatten=True, keep_metadata=True``: ``list[VariableState]``
  * ``flatten=False, keep_metadata=False``: nested ``dict`` of ``Array``
  * ``flatten=False, keep_metadata=True``: nested ``State`` of ``VariableState``

  The ``graphdef`` or the ``state`` can be replaced using the
  `replace` method::

    jax_model = jax_model.replace(graphdef=new_graphdef)
    jax_model = jax_model.replace(state=new_state)
  """

  _graphdef: graph.GraphDef[A]
  _state: tp.Any

  def __init__(
    self, node: A, flatten: bool = True, keep_metadata: bool = False
  ):
    """
    Args:
      node: The NNX object to be converted to a pytree.
      flatten: Whether to flatten the state or not, defaults to ``True``.
      keep_metadata: Whether to keep the metadata or not, defaults to ``False``.
    """
    graphdef: graph.GraphDef[A]
    state: tp.Any
    if flatten:
      if keep_metadata:
        graphdef, state = graph.flatten(node)
      else:
        graphdef, state = graph.flatten(
          node, with_paths=False, return_variables=False
        )
    else:
      graphdef, state = graph.split(node)
      if not keep_metadata:
        state = statelib.to_pure_dict(state)

    self._graphdef = graphdef
    self._state = state

  def __nnx_repr__(self):
    yield reprlib.Object(type(self))
    yield reprlib.Attr('graphdef', self._graphdef)
    yield reprlib.Attr('state', self._state)

  def __treescope_repr__(self, path, subtree_renderer):
    children = {
      'graphdef': self._graphdef,
      'state': self._state,
    }
    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes=children,
      path=path,
      subtree_renderer=subtree_renderer,
    )

  def __eq__(self, other):
    return (
      type(other) == type(self)
      and self._graphdef == other._graphdef
      and self._state == other._state
    )

  def __hash__(self):
    return hash((self._graphdef, self._state))

  @property
  def graphdef(self) -> graph.GraphDef[A]:
    return self._graphdef

  @property
  def state(self) -> tp.Any:
    return self._state

  def merge(self) -> A:
    node = graph.merge(self._graphdef, self._state)
    return node

  def replace(
    self,
    graphdef: graph.GraphDef[A] | Missing = MISSING,
    state: statelib.State | Missing = MISSING,
  ) -> Pytree[A]:
    pytree = object.__new__(Pytree)
    pytree._graphdef = (
      graphdef if not isinstance(graphdef, Missing) else self._graphdef
    )
    pytree._state = state if not isinstance(state, Missing) else self._state
    return pytree

  def __call__(self, *args, **kwargs):
    module = self.merge()
    out = module(*args, **kwargs)  # type: ignore
    return out

  def __getattr__(self, name: str) -> proxy_caller.CallableProxy:
    def _apply(accessor, *args, **kwargs):
      module = self.merge()
      fn = accessor(module)
      out = fn(*args, **kwargs)
      return out

    proxy = proxy_caller.CallableProxy(_apply)  # type: ignore[arg-type]
    return getattr(proxy, name)


def _pytree_flatten(pytree: Pytree[tp.Any]):
  return (pytree._graphdef, pytree._state), None


def _pytree_unflatten(_: None, children: tuple[graph.GraphDef[tp.Any], tp.Any]):
  pytree = object.__new__(Pytree)
  pytree._graphdef, pytree._state = children
  return pytree


jax.tree_util.register_pytree_node(Pytree, _pytree_flatten, _pytree_unflatten)
