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

from flax.nnx import filterlib
from flax.nnx import graphlib
from flax.nnx import pytreelib
from flax.nnx import variablelib
import jax
import jax.numpy as jnp

A = tp.TypeVar('A')


def _to_ema_param(node: tp.Any):
  def ema_param(x: variablelib.Variable) -> variablelib.Variable:
    ema_metadata = x.get_metadata()
    value = jnp.copy(x.get_value())
    return type(x)(value, **ema_metadata)

  return jax.tree.map(
      ema_param, node, is_leaf=lambda x: isinstance(x, variablelib.Variable)
  )


class EMA(pytreelib.Pytree):
  """Exponential Moving Average (EMA) of parameters.

  Maintains a shadow copy of model Variables that is updated as an
  exponentially weighted moving average on each call to :meth:`update`.
  This is commonly used to stabilize training and improve evaluation
  performance by applying the averaged parameters at inference time.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp
    >>> import optax
    ...
    >>> model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    >>> optimizer = nnx.Optimizer(model, optax.sgd(0.1))
    >>> ema = nnx.EMA(model, decay=0.9)
    >>> ema_model = ema.apply_to(model)
    ...
    >>> def loss_fn(model, x, y):
    ...   return jnp.mean((model(x) - y) ** 2)
    ...
    >>> @nnx.jit
    ... def train_step(model, optimizer, ema, x, y):
    ...   grads = nnx.grad(loss_fn)(model, x, y)
    ...   optimizer.update(model, grads)
    ...   ema.update(model)
    ...
    >>> @nnx.jit
    ... def eval_step(model, x, y):
    ...   return loss_fn(model, x, y)
    ...
    >>> x, y = jnp.ones((1, 2)), jnp.ones((1, 2))
    >>> train_step(model, optimizer, ema, x, y)
    >>> loss = eval_step(ema_model, x, y)

  In this example, ``ema.update`` computes the moving average and updates
  the internal state of ``ema``. ``ema.apply_to`` creates a new model
  instance (``ema_model``) that shares its Variables with ``ema``.
  Therefore, ``ema_model`` will automatically reflect the updates performed by
  ``ema.update`` and can be used directly in ``eval_step``.

  Attributes:
    decay: The decay rate for the exponential moving average.
    filter: The filter used to select which variables to track.
    params: A pytree of variables holding the current
      moving average values.
  """

  def __init__(
      self,
      params: tp.Any,
      decay: float,
      *,
      only: filterlib.Filter = ...,
      graph: bool | None = None,
  ):
    """Initializes the EMA module.

    Args:
      params: Any object, typically an NNX module/node, whose parameters
        will be tracked.
      decay: The decay rate for the moving average.
      only: A filter indicating which variables should be included in the
        EMA tracking. Defaults to matching everything. Note that EMA only
        tracks ``nnx.Variable`` instances.
      graph: If ``True``, uses graph-mode which supports the full NNX
        feature set including shared references. If ``False``, uses
        tree-mode which treats Modules as regular JAX pytrees, avoiding
        the overhead of the graph protocol. If ``None`` (default), the
        value is determined by the current ``nnx.set_graph_mode`` context.
    """
    only = filterlib.All(variablelib.Variable, only)
    self.graph = graph
    self.decay = decay
    self.filter = only
    self.params: graphlib.State = pytreelib.data(
        _to_ema_param(graphlib.state(params, only, graph=graph))
    )

  def update(self, updates: tp.Any) -> None:
    """Updates the EMA parameters towards the given new parameters.

    The update rule for each parameter is::

      ema = decay * ema + (1 - decay) * update

    Args:
      updates: The new parameters or module to blend into the current EMA.
        This should have the same structure as the ``params`` object passed
        during initialization.
    """
    def _update_ema(ema: variablelib.Variable, update: tp.Any) -> tp.Any:
      ema[...] = self.decay * ema + (1.0 - self.decay) * update

    jax.tree.map(
      _update_ema,
      self.params,
      graphlib.state(updates, self.filter, graph=self.graph),
      is_leaf=lambda x: isinstance(x, variablelib.Variable),
    )

  def apply_to(self, model: A) -> A:
    """Returns a view of the model using the EMA parameters.

    Constructs a new model instance with the same structure as ``model``
    but whose tracked parameters are replaced by their exponential moving
    average values. Non-tracked state (e.g. variables excluded by the
    ``only`` filter) is preserved from the original ``model``.

    This is typically used at evaluation time to obtain a model whose
    parameters reflect the smoothed training trajectory.

    Example usage::

      >>> from flax import nnx
      >>> import jax.numpy as jnp
      ...
      >>> model = nnx.Linear(2, 2, use_bias=False, rngs=nnx.Rngs(0))
      >>> ema = nnx.EMA(model, decay=0.9)
      >>> ema_model = ema.apply_to(model)
      >>> assert ema_model.kernel is ema.params.kernel

    Args:
      model: A model instance whose graph structure is used to build
        the output. The model should have the same structure as the
        ``params`` originally passed to :class:`EMA`.

    Returns:
      A new model of the same type as ``model`` with tracked parameters
      replaced by the current EMA values.
    """
    graphdef, state = graphlib.split(model, graph=self.graph)
    merged_state = graphlib.merge_state(state, self.params)
    return graphlib.merge(graphdef, merged_state)
