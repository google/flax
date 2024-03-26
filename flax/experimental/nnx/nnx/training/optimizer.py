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

from flax.experimental import nnx
from flax.experimental.nnx.nnx.variables import Variable

import optax

#TODO: add tests and docstrings

class OptState(Variable):
  """Wrapper class for Optimizer Variables."""
  pass

class Optimizer(nnx.Module):
  """Simple train state for the common case with a single Optax optimizer.

  Example usage::

    >>> import jax, jax.numpy as jnp
    >>> from flax.experimental import nnx
    >>> import optax

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
    ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.linear2(self.linear1(x))

    >>> x = jax.random.normal(jax.random.key(0), (1, 2))
    >>> y = jnp.ones((1, 4))

    >>> model = Model(nnx.Rngs(0))
    >>> tx = optax.adam(1e-3)
    >>> state = nnx.optimizers.Optimizer(model, tx)

    >>> loss_fn = lambda model: ((model(x)-y)**2).mean()
    >>> loss_fn(state.model)
    1.7055722
    >>> grads = nnx.grad(loss_fn, wrt=nnx.Param)(state.model)
    >>> state.update(grads=grads)
    >>> loss_fn(state.model)
    1.6925814

  Note that you can easily extend this class by subclassing it for storing
  additional data (e.g. adding metrics).

  Example usage::

    >>> class TrainState(nnx.optimizer.Optimizer):
    ...   def __init__(self, model, tx, metrics):
    ...     self.metrics = metrics
    ...     super().__init__(model, tx)
    ...   def update(self, *, grads, **updates):
    ...     self.metrics.update(**updates)
    ...     super().update(grads=grads)

    >>> metrics = nnx.metrics.Average()
    >>> state = TrainState(model, tx, metrics)

    >>> grads = nnx.grad(loss_fn, wrt=nnx.Param)(state.model)
    >>> state.update(grads=grads, values=loss_fn(state.model))
    >>> state.metrics.compute()
    1.6925814
    >>> state.update(grads=grads, values=loss_fn(state.model))
    >>> state.metrics.compute()
    1.68612

  For more exotic usecases (e.g. multiple optimizers) it's probably best to
  fork the class and modify it.

  Args:
    model: An NNX Module.
    tx: An Optax gradient transformation.
  """

  def __init__(
    self,
    model: nnx.Module,
    tx: optax.GradientTransformation,
  ):
    self.step = OptState(0)
    self.model = model
    self.tx = tx
    self.opt_state = tx.init(model.extract(nnx.Param))

  def apply_gradients(self, *, grads):
    """Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.

    Note that internally this function calls ``.tx.update()`` followed by a call
    to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.

    Args:
      grads: Gradients that have the same pytree structure as ``.params``.
      **kwargs: Additional dataclass attributes that should be ``.replace()``-ed.

    Returns:
      An updated instance of ``self`` with ``step`` incremented by one, ``params``
      and ``opt_state`` updated by applying ``grads``, and additional attributes
      replaced as specified by ``kwargs``.
    """
    params = self.model.extract(nnx.Param)

    updates, new_opt_state = self.tx.update(
      grads, self.opt_state, params
    )
    new_params= optax.apply_updates(params, updates)

    self.step.value += 1
    self.model.update(new_params)
    self.opt_state = new_opt_state

