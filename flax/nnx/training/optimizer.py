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

import functools
import typing as tp

import jax
import jax.numpy as jnp
import optax

from flax import nnx
from flax.nnx import filterlib
from flax.nnx.pytreelib import Pytree
from flax.nnx.variablelib import Variable

M = tp.TypeVar('M', bound=nnx.Module)
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])

class OptState(Variable):
  """Any optimizer state"""

  pass


class OptArray(OptState):
  """Optimizer state for an array."""

  pass


class OptVariable(OptState):
  """Optimizer state for a Variable."""

  pass


def to_opt_state(tree):
  def _to_opt_state(x):
    if isinstance(x, Variable):
      opt_state = OptVariable(x.get_value(), **x.get_metadata())  # type: ignore
    else:
      opt_state = OptArray(x)
    return opt_state

  tree = jax.tree.map(
    _to_opt_state,
    tree,
    is_leaf=lambda x: isinstance(x, Variable),
  )
  return tree

class _Missing:
  pass

MISSING = _Missing()

def _check_grads_arg_passed(f: F) -> F:
  @functools.wraps(f)
  def _check_grads_wrapper(self, model, grads=MISSING, **kwargs):
    if isinstance(grads, _Missing):
      raise TypeError(
        'Missing required argument `grads`. As of Flax 0.11.0 update requires both (model, grads) arguments '
        'to be passed. If you want to keep the previous use nnx.ModelAndOptimizer instead of nnx.Optimizer.'
      )
    return f(self, model, grads, **kwargs)
  return _check_grads_wrapper # type: ignore

def _check_wrt_arg_passed(f: F) -> F:
  @functools.wraps(f)
  def _check_wrt_wrapper(*args, wrt=MISSING, **kwargs):
    if isinstance(wrt, _Missing):
      raise TypeError(
        'Missing required argument `wrt`. As of Flax 0.11.0 the `wrt` argument is required, '
        'if you want to keep the previous use nnx.ModelAndOptimizer instead of nnx.Optimizer.'
      )
    return f(*args, wrt=wrt, **kwargs)
  return _check_wrt_wrapper  # type: ignore

class Optimizer(Pytree, tp.Generic[M]):
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
    >>> optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    ...
    >>> loss_fn = lambda model: ((model(x) - y) ** 2).mean()
    >>> loss_fn(model)
    Array(2.3359997, dtype=float32)
    >>> grads = nnx.grad(loss_fn)(model)
    >>> optimizer.update(model, grads)
    >>> loss_fn(model)
    Array(2.310461, dtype=float32)

  Attributes:
    step: An ``OptState`` :class:`Variable` that tracks the step count.
    tx: An Optax gradient transformation.
    opt_state: The Optax optimizer state.
  """

  @_check_wrt_arg_passed
  def __init__(
    self,
    model: M,
    tx: optax.GradientTransformation,
    *,
    wrt: filterlib.Filter,  # type: ignore
  ):
    """
    Instantiate the class and wrap the :class:`Module` and Optax gradient
    transformation. Instantiate the optimizer state to keep track of
    :class:`Variable` types specified in ``wrt``. Set the step count to 0.

    Args:
      model: An NNX Module.
      tx: An Optax gradient transformation.
      wrt: filter to specify for which :class:`Variable`'s to keep
        track of in the optimizer state. These should be the :class:`Variable`'s
        that you plan on updating; i.e. this argument value should match the
        ``wrt``  argument passed to the ``nnx.grad`` call that will generate the
        gradients that will be passed into the ``grads`` argument of the
        :func:`update` method. The filter should match the filter used in nnx.grad.
    """
    if isinstance(wrt, _Missing):
      raise TypeError(
        'Missing required argument `wrt`. As of Flax 0.11.0 the `wrt` argument is required, '
        'if you want to keep the previous use nnx.ModelAndOptimizer instead of nnx.Optimizer.'
      )
      wrt = nnx.Param
    self.step = OptState(jnp.array(0, dtype=jnp.uint32))
    self.tx = tx
    self.opt_state = nnx.data(
      to_opt_state(tx.init(nnx.state(model, wrt)))
    )
    self.wrt = wrt

  if not tp.TYPE_CHECKING:
    def __getattribute__(self, name: str) -> tp.Any:
      if name == 'model' and name not in vars(self):
        raise AttributeError(
          f"{type(self).__name__} does not have attribute 'model' since Flax 0.11.0. "
          "To keep the previous behavior, use nnx.ModelAndOptimizer instead of nnx.Optimizer."
        )
      return super().__getattribute__(name)

  @_check_grads_arg_passed
  def update(self, model: M, grads, /, **kwargs):
    """Updates the optimizer state and model parameters given the gradients.

    Example::

      >>> from flax import nnx
      >>> import jax, jax.numpy as jnp
      >>> import optax
      ...
      >>> class Model(nnx.Module):
      ...   def __init__(self, rngs):
      ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
      ...     self.count = nnx.Variable(jnp.array(0))
      ...
      ...   def __call__(self, x):
      ...     self.count[...] += 1
      ...     return self.linear(x)
      ...
      >>> model = Model(rngs=nnx.Rngs(0))
      ...
      >>> loss_fn = lambda model, x, y: ((model(x) - y) ** 2).mean()
      >>> optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
      >>> grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(
      ...   model, jnp.ones((1, 2)), jnp.ones((1, 3))
      ... )
      >>> optimizer.update(model, grads)

    Note that internally this function calls ``.tx.update()`` followed by a call
    to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.

    Args:
      grads: the gradients derived from ``nnx.grad``.
      **kwargs: additional keyword arguments passed to the tx.update, to support
      ``GradientTransformationExtraArgs``, such as ``optax.scale_by_backtracking_linesearch``.
    """
    param_arrays = nnx.pure(nnx.state(model, self.wrt))
    grad_arrays = nnx.pure(nnx.state(grads, self.wrt))
    opt_state_arrays = nnx.pure(self.opt_state)
    kwargs_arrays = nnx.pure(kwargs)

    updates, new_opt_state = self.tx.update(
      grad_arrays, opt_state_arrays, param_arrays, **kwargs_arrays
    )
    new_params = optax.apply_updates(param_arrays, updates)

    nnx.update(model, new_params)
    nnx.update(self.opt_state, nnx.state(new_opt_state))
    self.step[...] += 1

class ModelAndOptimizer(Optimizer[M]):
  """A convenience class that combines a model and an optimizer.

  This class is deprecated and will be removed in a future release.
  Use :class:`Optimizer` instead.
  """

  def __init__(self, model: M, tx: optax.GradientTransformation, *, wrt: filterlib.Filter = nnx.Param):
    super().__init__(model, tx, wrt=wrt)
    self.model = model

  def update(self, grads, /, **kwargs): # type: ignore
    return super().update(self.model, grads, **kwargs)
