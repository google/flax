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

import jax.numpy as jnp
import optax

from flax import nnx
from flax.nnx.nnx import filterlib, graph
from flax.nnx.nnx.object import Object
from flax.nnx.nnx.variables import Variable

# TODO: add tests and docstrings


class OptState(Variable):
  """Wrapper class for Optimizer Variables."""

  pass


class Optimizer(Object):
  """Simple train state for the common case with a single Optax optimizer.

  Example usage::

    >>> import jax, jax.numpy as jnp
    >>> from flax import nnx
    >>> import optax
    ...
    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
    ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.linear2(self.linear1(x))
    ...
    >>> x = jax.random.normal(jax.random.key(0), (1, 2))
    >>> y = jnp.ones((1, 4))
    ...
    >>> model = Model(nnx.Rngs(0))
    >>> tx = optax.adam(1e-3)
    >>> state = nnx.Optimizer(model, tx)
    ...
    >>> loss_fn = lambda model: ((model(x) - y) ** 2).mean()
    >>> loss_fn(model)
    Array(1.7055722, dtype=float32)
    >>> grads = nnx.grad(loss_fn, wrt=nnx.Param)(state.model)
    >>> state.update(grads)
    >>> loss_fn(model)
    Array(1.6925814, dtype=float32)

  Note that you can easily extend this class by subclassing it for storing
  additional data (e.g. adding metrics).

  Example usage::

    >>> class TrainState(nnx.Optimizer):
    ...   def __init__(self, model, tx, metrics):
    ...     self.metrics = metrics
    ...     super().__init__(model, tx)
    ...   def update(self, *, grads, **updates):
    ...     self.metrics.update(**updates)
    ...     super().update(grads)
    ...
    >>> metrics = nnx.metrics.Average()
    >>> state = TrainState(model, tx, metrics)
    ...
    >>> grads = nnx.grad(loss_fn, wrt=nnx.Param)(state.model)
    >>> state.update(grads=grads, values=loss_fn(state.model))
    >>> state.metrics.compute()
    Array(1.6925814, dtype=float32)
    >>> state.update(grads=grads, values=loss_fn(state.model))
    >>> state.metrics.compute()
    Array(1.68612, dtype=float32)

  For more exotic usecases (e.g. multiple optimizers) it's probably best to
  fork the class and modify it.

  Attributes:
    step: An ``OptState`` :class:`Variable` that tracks the step count.
    model: The wrapped :class:`Module`.
    tx: An Optax gradient transformation.
    opt_state: The Optax optimizer state.
  """

  def __init__(
    self,
    model: nnx.Module,
    tx: optax.GradientTransformation,
    wrt: filterlib.Filter = nnx.Param,
  ):
    """
    Instantiate the class and wrap the :class:`Module` and Optax gradient
    transformation. Instantiate the optimizer state to keep track of
    :class:`Variable` types specified in ``wrt``. Set the step count to 0.

    Args:
      model: An NNX Module.
      tx: An Optax gradient transformation.
      wrt: optional argument to filter for which :class:`Variable`'s to keep
        track of in the optimizer state. These should be the :class:`Variable`'s
        that you plan on updating; i.e. this argument value should match the
        ``wrt``  argument passed to the ``nnx.grad`` call that will generate the
        gradients that will be passed into the ``grads`` argument of the
        :func:`update` method.
    """
    self.step = OptState(jnp.array(0, dtype=jnp.uint32))
    self.model = model
    self.tx = tx
    self.opt_state = tx.init(nnx.state(model, wrt))
    self.wrt = wrt

  def split(self, *filters: filterlib.Filter):
    return graph.split(self, *filters)

  def update(self, grads):
    """Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.
    The ``grads`` must be derived from ``nnx.grad(..., wrt=self.wrt)``, where the
    gradients are with respect to the same :class:`Variable` types as defined in
    ``self.wrt`` during instantiation of this ``Optimizer``. For example::

      >>> from flax import nnx
      >>> import jax, jax.numpy as jnp
      >>> import optax

      >>> class CustomVariable(nnx.Variable):
      ...   pass

      >>> class Model(nnx.Module):
      ...   def __init__(self, rngs):
      ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
      ...     self.custom_variable = CustomVariable(jnp.ones((1, 3)))
      ...   def __call__(self, x):
      ...     return self.linear(x) + self.custom_variable
      >>> model = Model(rngs=nnx.Rngs(0))
      >>> jax.tree.map(jnp.shape, nnx.state(model))
      State({
        'custom_variable': VariableState(
          type=CustomVariable,
          value=(1, 3)
        ),
        'linear': {
          'bias': VariableState(
            type=Param,
            value=(3,)
          ),
          'kernel': VariableState(
            type=Param,
            value=(2, 3)
          )
        }
      })

      >>> # update:
      >>> # - only Linear layer parameters
      >>> # - only CustomVariable parameters
      >>> # - both Linear layer and CustomVariable parameters
      >>> loss_fn = lambda model, x, y: ((model(x) - y) ** 2).mean()
      >>> for variable in (nnx.Param, CustomVariable, (nnx.Param, CustomVariable)):
      ...   # make sure `wrt` arguments match for `nnx.Optimizer` and `nnx.grad`
      ...   state = nnx.Optimizer(model, optax.adam(1e-3), wrt=variable)
      ...   grads = nnx.grad(loss_fn, wrt=variable)(
      ...     state.model, jnp.ones((1, 2)), jnp.ones((1, 3))
      ...   )
      ...   state.update(grads=grads)

    Note that internally this function calls ``.tx.update()`` followed by a call
    to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.

    Args:
      grads: the gradients derived from ``nnx.grad``.
    """
    state = nnx.state(self.model, self.wrt)

    updates, new_opt_state = self.tx.update(grads, self.opt_state, state)
    new_params = optax.apply_updates(state, updates)
    assert isinstance(new_params, nnx.State)

    self.step.value += 1
    nnx.update(self.model, new_params)
    self.opt_state = new_opt_state
