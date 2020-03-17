# Lint as: python3

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flax Optimizer api.

Flax optimizers are defined using the OptimizerDef class which specifies the
initialization and gradient application logic.
Creating an optimizer using the `create` method will result in an instance of
the `Optimizer` class which encapsulates the optimization target and state.

Example of constructing an optimizer for a model::

  from flax import optim
  optimizer_def = optim.GradientDescent(learning_rate=0.1)
  optimizer = optimizer_def.create(model)

The optimizer is then used in a training step as follows::

  def train_step(optimizer, data):
    def loss_fn(model):
      y = model(data)
      loss = ... # compute the loss
      aux = ... # compute auxiliary outputs (eg. training metrics)
      return loss, aux
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grad = grad_fn(optimizer.target)
    new_optimizer = optimizer.apply_gradient(grad)
    return new_optimizer, loss, aux


Distributed training only requires a few extra additions::

  from flax import optim
  optimizer_def = optim.GradientDescent(learning_rate=0.1)
  optimizer = optimizer_def.create(model)
  optimizer = jax_utils.replicate(optimizer)

  def train_step(optimizer, data):
    def loss_fn(model):
      y = model(data)
      loss = ... # compute the loss
      aux = ... # compute auxiliary outputs (eg. training metrics)
      return loss, aux
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grad = grad_fn(optimizer.target)
    grad = jax.lax.pmean(grad, 'batch')
    new_optimizer = optimizer.apply_gradient(grad)
    return new_optimizer, loss, aux

  distributed_train_step = jax.pmap(train_step, axis_name='batch')

"""

import abc
from typing import Any
import warnings

from . import jax_utils
from . import serialization
from . import struct
from . import traverse_util

import jax
from jax import lax
import jax.numpy as jnp

from .nn import base

import numpy as onp


@struct.dataclass
class OptimizerState:
  step: int
  param_states: Any


class OptimizerDef:
  """Base class for optimizers."""

  def __init__(self, hyper_params):
    self.hyper_params = hyper_params

  @abc.abstractmethod
  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    """Apply a gradient for a single parameter.

    Args:
      step: the current step of the optimizer.
      hyper_params: a named tuple of hyper parameters.
      param: the parameter that should be updated.
      state: a named tuple containing the state for this parameter
      grad: the gradient tensor for the parameter.
    Returns:
      A tuple containing the new parameter and the new state.
    """
    pass

  @abc.abstractmethod
  def init_param_state(self, param):
    """Initializes the state for a parameter.

    Args:
      param: the parameter for which to initialize the state.
    Returns:
      A named tuple containing the initial optimization state for the parameter.
    """
    pass

  def apply_gradient(self, hyper_params, params, state, grads):
    """Applies a gradient for a set of parameters.

    Args:
      hyper_params: a named tuple of hyper parameters.
      params: the parameters that should be updated.
      state: a named tuple containing the state of the optimizer
      grads: the gradient tensors for the parameters.
    Returns:
      A tuple containing the new parameters and the new optimizer state.
    """
    step = state.step
    params_flat, treedef = jax.tree_flatten(params)
    states_flat = treedef.flatten_up_to(state.param_states)
    grads_flat = treedef.flatten_up_to(grads)
    out = [self.apply_param_gradient(step, hyper_params, param, state, grad)
           for param, state, grad in zip(params_flat, states_flat, grads_flat)]

    new_params_flat, new_states_flat = list(zip(*out))
    new_params = jax.tree_unflatten(treedef, new_params_flat)
    new_param_states = jax.tree_unflatten(treedef, new_states_flat)
    new_state = OptimizerState(step + 1, new_param_states)
    return new_params, new_state

  def init_state(self, params):
    param_states = jax.tree_map(self.init_param_state, params)
    state = OptimizerState(0, param_states)
    return state

  def update_hyper_params(self, **hyper_param_overrides):
    """Updates the hyper parameters with a set of overrides.

    This method is called from Optimizer apply_gradient to create the
    hyper parameters for a specific optimization step.

    Args:
      **hyper_param_overrides: the hyper parameters updates
        will override the defaults specified in the `OptimizerDef`.
        Pass `hyper_params=...` to replace all hyper parameters.
    Returns:
      The new hyper parameters.
    """
    hp = hyper_param_overrides.pop('hyper_params', self.hyper_params)
    if hyper_param_overrides:
      hp = hp.replace(**hyper_param_overrides)
    return hp

  def create(self, target, focus=None):
    """Creates a new optimizer for the given target.

    Args:
      target: the object to be optimized. This will typically be
        an instance of `flax.nn.Model`.
      focus: a `flax.traverse_util.Traversal` that selects which subset of
        the target is optimized.
    Returns:
      An instance of `Optimizer`.
    """
    opt_def = self
    if focus:
      opt_def = MultiOptimizer((focus, opt_def))
    state = opt_def.init_state(target)
    return Optimizer(opt_def, state, target)

  def state_dict(self, target, state):
    return serialization.to_state_dict({
        'target': serialization.to_state_dict(target),
        'state': serialization.to_state_dict(state)
    })

  def restore_state(self, opt_target, opt_state, state_dict):
    """Restore the optimizer target and state from the state dict.

    This function accepts the current optimizer target and state. This
    lets us know the exact structure of the optimizer target and state,
    as well as lets us add assertions that shapes and dtypes don't change.

    In practice, no values in `opt_target` and `opt_state` are actually
    used. Only the tree structure, shapes and types.

    Args:
      opt_target: the optimizer target.
      opt_state: the optimizer state.
      state_dict: the state dict containing the desired new state of the
                  optimizer.
    Returns:
      a tuple of the optimizer target and state with the restored values from
      the state dict.
    """

    opt_target = serialization.from_state_dict(opt_target, state_dict['target'])
    opt_state = serialization.from_state_dict(opt_state, state_dict['state'])
    return opt_target, opt_state


class _NoAux:
  """Placeholder used to indicate a lack of auxilairy outputs."""
  pass


@struct.dataclass
class Optimizer:
  """Wraps an optimizer with its hyper_params, state, and model parameters."""

  optimizer_def: OptimizerDef = struct.field(pytree_node=False)
  state: Any
  target: Any

  def apply_gradient(self, grads, **hyper_param_overrides):
    """Applies a pytree of gradients to the target.

    Args:
      grads: A pytree of gradients.
      **hyper_param_overrides: the hyper parameters passed to apply_gradient
        will override the defaults specified in the `OptimizerDef`.
        Pass `hyper_params=...` to replace all hyper parameters.
    Returns:
      A new optimizer with the updated target and state.
    """
    hyper_params = self.optimizer_def.update_hyper_params(
        **hyper_param_overrides)
    new_target, new_state = self.optimizer_def.apply_gradient(
        hyper_params, self.target, self.state, grads)
    return self.replace(target=new_target, state=new_state)

  def compute_gradient(self, loss_fn):
    """Computes gradient of loss_fn.

    Args:
      loss_fn: a function that receives the target and returns a loss or a
        tuple of the loss and auxiliary outputs.
    Returns:
      A tuple consisting of the loss, auxiliary outputs if any,
        and a list of gradient.
    """
    def loss_wrapper(target):
      loss_and_aux = loss_fn(target)
      if isinstance(loss_and_aux, jnp.ndarray):
        return loss_and_aux, _NoAux
      else:
        return loss_and_aux
    grad_fn = jax.value_and_grad(loss_wrapper, has_aux=True)
    (loss, aux), grad = grad_fn(self.target)
    if aux is _NoAux:
      return loss, grad
    else:
      return loss, aux, grad
  compute_gradients = compute_gradient

  def optimize(self, loss_fn, **hyper_param_overrides):
    """Optimizes the target with respect to a loss function.

    DEPRECATION WARNING:
    optimize() is deprecated.
    Use jax.grad() or jax.value_and_grad() and apply_gradient() instead.

    Args:
      loss_fn:  function that receives the target and returns a loss or a
        tuple of the loss and auxiliary outputs.
      **hyper_param_overrides: the hyper parameters passed to apply_gradient
        will override the defaults specified in the `OptimizerDef`.
        Pass `hyper_params=...` to replace all hyper parameters.
    Returns:
      A tuple consisting of the new optimizer, the loss,
        and the auxiliary outputs if any.
    """
    warnings.warn('optimize() will be removed soon.'
                  ' Use jax.grad() or jax.value_and_grad()'
                  'and apply_gradient() instead.',
                  DeprecationWarning)

    output_and_grad = self.compute_gradient(loss_fn)
    grad = output_and_grad[-1]
    optimizer = self.apply_gradient(grad, **hyper_param_overrides)
    return (optimizer,) + output_and_grad[:-1]

  def replicate(self, devices=None, axis_name='batch'):
    """Replicates an optimizer for data parallel training.

    A replicated optimizer will automatically average the gradients across
      devices. For this to work correctly the optimize method should be called
      within the context of a `jax.pmap` call with the correct axis_name.
    Args:
      devices: an optional list of devices defining which devices this optimizer
        is replicated to (default: all local devices).
      axis_name: the axis_name used for gradient averaging across devices.
    Returns:
      The replicated optimizer.
    """
    if devices is None:
      devices = jax.local_devices()
    optimizer_def = ReplicatedOptimizer(self.optimizer_def, devices, axis_name)
    optimizer = jax_utils.replicate(self, devices=devices)
    return optimizer.replace(optimizer_def=optimizer_def)

  def unreplicate(self):
    """Un-replicates an optimizer.

    This will create a new optimizer with the target and state of the first
      device this optimizer was replicated to. After this call the optimizer
      and the target can be used outside of a `jax.pmap` call.

    Returns:
      The optimizer that is no longer replicated.
    """
    if not isinstance(self.optimizer_def, ReplicatedOptimizer):
      raise ValueError('Cannot unreplicate an optimizer '
                       'that is not replicated.')
    optimizer_def = self.optimizer_def.optimizer_def
    optimizer = jax_utils.unreplicate(self)
    return optimizer.replace(optimizer_def=optimizer_def)

  def state_dict(self):
    return self.optimizer_def.state_dict(self.target, self.state)

  def restore_state(self, state):
    target, state = self.optimizer_def.restore_state(
        self.target, self.state, state)
    return self.replace(target=target, state=state)


# Optimizer serialization is handled by the state_dict and restore_dict methods
# of the OptimizerDef. Currently, this is used to store only a single copy of
# a replicated optimizer.
serialization.register_serialization_state(
    Optimizer, Optimizer.state_dict, Optimizer.restore_state,
    override=True)


class ReplicatedOptimizer(OptimizerDef):
  """Data parallel optimizer.

  DEPRECATION WARNING:
  ReplicatedOptimizer will be removed soon.
  Use `jax_utils.replicate(optimizer)` and `lax.pmean(grad)` to explicitly
  control the replication of the the optimizer and the cross replica averaging
  over gradients, respectively.
  """

  def __init__(self, optimizer_def, devices=None, axis_name='batch'):
    super().__init__(optimizer_def.hyper_params)
    if devices is None:
      devices = jax.local_devices()
    self.optimizer_def = optimizer_def
    self.devices = devices
    self.axis_name = axis_name

  def init_state(self, params):
    return self.optimizer_def.init_state(params)

  def _cross_replica_mean(self, grad):
    axis_size = jax.lax.psum(1, axis_name=self.axis_name)
    return jax.lax.psum(grad, axis_name=self.axis_name) / axis_size

  def apply_gradient(self, hyper_params, params, state, grads):
    grads = jax.tree_map(self._cross_replica_mean, grads)
    return self.optimizer_def.apply_gradient(hyper_params, params, state, grads)

  def update_hyper_params(self, **hyper_param_overrides):
    return self.optimizer_def.update_hyper_params(**hyper_param_overrides)

  def state_dict(self, target, state):
    state_dict = self.optimizer_def.state_dict(target, state)
    # only the first copy of the parameters and optimizer state are stored.
    state_dict = jax.tree_map(lambda x: x[0], state_dict)
    return state_dict

  def restore_state(self, target, opt_state, state_dict):
    # replicate the parameters and state to all devices.
    state_dict = jax_utils.replicate(state_dict, devices=self.devices)
    return self.optimizer_def.restore_state(target, opt_state, state_dict)


class MultiOptimizer(OptimizerDef):
  """Combine a set of optimizers by applying each to a subset of the parameters."""

  def __init__(self, *traversals_and_optimizers):
    """Create a new MultiOptimizer.

    A MultiOptimizer is useful when separate optimizer algorithms should be
    applied to various subsets of the model parameters.

    Example::

      kernels = optim.ModelParamTraversal(lambda path, _: 'kernel' in path)
      biases = optim.ModelParamTraversal(lambda path, _: 'bias' in path)
      kernel_opt = optim.Momentum(learning_rate=0.01)
      bias_opt = optim.Momentum(learning_rate=0.1)
      opt_def = MultiOptimizer((kernels, kernel_opt), (biases, bias_opt))
      optimizer = opt_def.create(model)


    Args:
      *traversals_and_optimizers: pairs of flax.traverse_util.Traversal and
      `flax.optim.OptimizerDef` instances.
    """
    traversals, sub_optimizers = zip(*traversals_and_optimizers)
    hyper_params = [opt.hyper_params for opt in sub_optimizers]
    super().__init__(hyper_params)
    self.traversals = traversals
    self.sub_optimizers = sub_optimizers

  def init_state(self, params):
    sub_states = []
    for traversal, opt in zip(self.traversals, self.sub_optimizers):
      params_t = list(traversal.iterate(params))
      state = opt.init_state(params_t)
      sub_states.append(state)
    return sub_states

  def apply_gradient(self, hyper_params, params, states, grads):
    new_params = params
    new_states = []
    it = zip(self.traversals, self.sub_optimizers, hyper_params, states)
    for focus, opt, hp, s in it:
      p = list(focus.iterate(params))
      g = list(focus.iterate(grads))
      new_p, new_s = opt.apply_gradient(hp, p, s, g)
      new_params = focus.set(new_p, new_params)
      new_states.append(new_s)
    return new_params, new_states

  def update_hyper_params(self, **hyper_param_overrides):
    """Updates the hyper parameters with a set of overrides.

    This method is called from Optimizer apply_gradient to create the
    hyper parameters for a specific optimization step.
    MultiOptimizer will apply the overrides for each sub optimizer.

    Args:
      **hyper_param_overrides: the hyper parameters updates
        will override the defaults specified in the `OptimizerDef`.
        Pass `hyper_params=...` to replace all hyper parameters.
    Returns:
      The new hyper parameters.
    """
    hps = hyper_param_overrides.pop('hyper_params', self.hyper_params)
    if hyper_param_overrides:
      hps = [hp.replace(**hyper_param_overrides) for hp in hps]
    return hps


def _sorted_items(x):
  """Returns items of a dict ordered by keys."""
  return sorted(x.items(), key=lambda x: x[0])


class ModelParamTraversal(traverse_util.Traversal):
  """Select model parameters using a name filter."""

  def __init__(self, filter_fn):
    """Constructor a new ModelParamTraversal.

    Args:
      filter_fn: a function that takes a parameters full name and its value and
        returns whether this parameter should be selected or not. The name of a
        parameter is determined by the module hierarchy and the parameter name
        (for example: '/module/sub_module/parameter_name').
    """
    self._filter_fn = filter_fn

  @staticmethod
  def _check_inputs(inputs):
    if not isinstance(inputs, base.Model):
      raise ValueError(
          'ModelParamTraversal can only traverse a flax Model instance.')

  def _iterate(self, x, path=''):
    if not isinstance(x, dict):
      # x is a leaf
      if self._filter_fn(path, x):
        yield x
    else:
      for key, value in _sorted_items(x):
        yield from self._iterate(value, '{}/{}'.format(path, key))

  def _update(self, fn, x, path=''):
    if not isinstance(x, dict):
      if self._filter_fn(path, x):
        return fn(x)
      else:
        return x
    else:
      new_x = {}
      for key, value in _sorted_items(x):
        new_x[key] = self._update(fn, value, '{}/{}'.format(path, key))
      return new_x

  def iterate(self, inputs):
    self._check_inputs(inputs)
    yield from self._iterate(inputs.params)

  def update(self, fn, inputs):
    self._check_inputs(inputs)
    new_params = self._update(fn, inputs.params)
    return inputs.replace(params=new_params)


@struct.dataclass
class _GradientDescentHyperParams:
  learning_rate: onp.ndarray


class GradientDescent(OptimizerDef):
  """Gradient descent optimizer."""

  def __init__(self, learning_rate=None):
    """Constructor for the GradientDescent optimizer.

    Args:
      learning_rate: the step size used to update the parameters.
    """
    hyper_params = _GradientDescentHyperParams(learning_rate)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    return ()

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    del step
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    new_param = param - hyper_params.learning_rate * grad
    return new_param, state


@struct.dataclass
class _MomentumHyperParams:
  learning_rate: onp.ndarray
  beta: onp.ndarray
  weight_decay: onp.ndarray
  nesterov: bool


@struct.dataclass
class _MomentumParamState:
  momentum: onp.ndarray


class Momentum(OptimizerDef):
  """Momentum optimizer."""

  def __init__(self, learning_rate=None, beta=0.9, weight_decay=0,
               nesterov=False):
    """Constructor for the Momentum optimizer.

    Args:
      learning_rate: the step size used to update the parameters.
      beta: the coefficient used for the moving average of the
        gradient (default: 0.9).
      weight_decay: weight decay coefficient to apply (default: 0).
      nesterov: whether to use Nesterov momentum (default: False).
    """

    hyper_params = _MomentumHyperParams(
        learning_rate, beta, weight_decay, nesterov)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    return _MomentumParamState(jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    del step
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    if hyper_params.weight_decay != 0:
      grad += hyper_params.weight_decay * param
    momentum = state.momentum
    new_momentum = hyper_params.beta * momentum + grad
    if hyper_params.nesterov:
      d_p = grad + hyper_params.beta * new_momentum
    else:
      d_p = new_momentum
    new_param = param - hyper_params.learning_rate * d_p
    new_state = _MomentumParamState(new_momentum)
    return new_param, new_state


@struct.dataclass
class _AdamHyperParams:
  learning_rate: onp.ndarray
  beta1: onp.ndarray
  beta2: onp.ndarray
  eps: onp.ndarray
  weight_decay: onp.ndarray


@struct.dataclass
class _AdamParamState:
  grad_ema: onp.ndarray
  grad_sq_ema: onp.ndarray


class Adam(OptimizerDef):
  """Adam optimizer."""

  def __init__(self,
               learning_rate=None,
               beta1=0.9,
               beta2=0.999,
               eps=1e-8,
               weight_decay=0.0):
    """Constructor for the Adam optimizer.

    Args:
      learning_rate: the step size used to update the parameters.
      beta1: the coefficient used for the moving average of the
        gradient (default: 0.9).
      beta2: the coefficient used for the moving average of the
        gradient magnitude (default: 0.999).
      eps: the term added to the gradient magnitude estimate for
        numerical stability.
      weight_decay: AdamW style weight decay rate
        (relative to learning rate).
    """
    hyper_params = _AdamHyperParams(learning_rate, beta1, beta2, eps,
                                    weight_decay)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    return _AdamParamState(jnp.zeros_like(param), jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    beta1 = hyper_params.beta1
    beta2 = hyper_params.beta2
    weight_decay = hyper_params.weight_decay
    grad_sq = lax.square(grad)
    grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
    grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

    # bias correction
    t = step + 1.
    grad_ema_corr = grad_ema / (1 - beta1 ** t)
    grad_sq_ema_corr = grad_sq_ema / (1 - beta2 ** t)

    denom = jnp.sqrt(grad_sq_ema_corr) + hyper_params.eps
    new_param = param - hyper_params.learning_rate * grad_ema_corr / denom
    if weight_decay != 0.0:
      new_param -= hyper_params.learning_rate * weight_decay * param
    new_state = _AdamParamState(grad_ema, grad_sq_ema)
    return new_param, new_state


@struct.dataclass
class _LARSHyperParams:
  learning_rate: onp.ndarray
  beta: onp.ndarray
  weight_decay: onp.ndarray
  trust_coefficient: onp.ndarray
  eps: onp.ndarray
  nesterov: bool


@struct.dataclass
class _LARSParamState:
  momentum: onp.ndarray


class LARS(OptimizerDef):
  """Layerwise adaptive rate scaling (LARS) optimizer."""

  def __init__(self, learning_rate=None, beta=0.9, weight_decay=0,
               trust_coefficient=0.001, eps=0, nesterov=False):
    """Constructor for the LARS optimizer.

    Args:
      learning_rate: the step size used to update the parameters.
      beta: the coefficient used for the moving average of the
        gradient (default: 0.9).
      weight_decay: weight decay coefficient to apply
      trust_coefficient: coefficient for trust ratio computation
        (default: 0.001).
      eps: epsilon used for trust ratio computation (default: no epsilon).
      nesterov: whether to use Nesterov momentum (default: False).
    """

    hyper_params = _LARSHyperParams(
        learning_rate, beta, weight_decay, trust_coefficient, eps, nesterov)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    return _LARSParamState(jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    del step
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'

    param_norm = jnp.linalg.norm(param)
    grad_norm = jnp.linalg.norm(grad)
    trust_ratio = hyper_params.trust_coefficient * param_norm / (
        grad_norm + hyper_params.weight_decay * param_norm + hyper_params.eps)
    clipped_trust_ratio = jnp.where(
        param_norm + grad_norm > 0., trust_ratio, 1.)
    scaled_lr = hyper_params.learning_rate * clipped_trust_ratio
    if hyper_params.weight_decay != 0:
      grad += hyper_params.weight_decay * param

    scaled_grad = scaled_lr * grad
    momentum = state.momentum
    new_momentum = hyper_params.beta * momentum + scaled_grad
    if hyper_params.nesterov:
      d_p = scaled_grad + hyper_params.beta * new_momentum
    else:
      d_p = new_momentum
    new_param = param - d_p
    new_state = _LARSParamState(new_momentum)
    return new_param, new_state


@struct.dataclass
class _WeightNormHyperParams:
  inner: Any
  wn_decay: onp.ndarray
  wn_eps: onp.ndarray


@struct.dataclass
class _WeightNormParamState:
  direction_state: Any
  scale_state: Any
  mult: onp.ndarray


class WeightNorm(OptimizerDef):
  """Adds weight normalization to an optimizer def.

  See https://arxiv.org/abs/1602.07868
  """

  def __init__(self, wrapped_optimizer, wn_decay=0, wn_eps=1e-8):
    """Constructor for a WeightNorm optimizer.

    Weight vectors are decomposed as w = g * v/||v||_2, for scalar
    scale parameter g, and raw weight vector v. The original optimizer is then
    applied to the (g,v) parameterization and the updated parameters are
    transformed back to w-space, i.e.
    w,state --> (g,v) --(original optimizer)--> (g',v') --> w',state'

    We assume the output axis of any kernel matrix is the last one,
    as per the Tensorflow convention.

    Args:
      wrapped_optimizer: another OptimizerDef
      wn_decay: apply l2 decay to the unnoralized weight vector
      wn_eps: additive constant for stability of
        the normalization (default: 1e-8).
    """
    hps = _WeightNormHyperParams(
        wrapped_optimizer.hyper_params, wn_decay, wn_eps)
    super().__init__(hps)
    self.wrapped_optimizer = wrapped_optimizer

  def update_hyper_params(self, **hyper_param_overrides):
    decay = hyper_param_overrides.pop('wn_decay', self.hyper_params.wn_decay)
    eps = hyper_param_overrides.pop('wn_eps', self.hyper_params.wn_eps)
    inner = self.wrapped_optimizer.update_hyper_params(
        **hyper_param_overrides)
    return self.hyper_params.replace(inner=inner, wn_decay=decay, wn_eps=eps)

  def init_state(self, params):
    leaves, treedef = jax.tree_flatten(params)
    directions, scales = zip(*(self._split_param(p) for p in leaves))
    directions = treedef.unflatten(directions)
    scales = treedef.unflatten(scales)
    wn_params = {'direction': directions, 'scale': scales}
    state = self.wrapped_optimizer.init_state(wn_params)
    direction_state = state.param_states['direction']
    scale_state = state.param_states['scale']
    param_states = jax.tree_multimap(
        lambda _, *args: _WeightNormParamState(*args),
        params, direction_state, scale_state, scales)
    return state.replace(param_states=param_states)

  def apply_gradient(self, hyper_params, params, state, grads):
    p_leaves, treedef = jax.tree_flatten(params)
    s_leaves = treedef.flatten_up_to(state.param_states)
    g_leaves = treedef.flatten_up_to(grads)
    split_grads = zip(*(self._split_grad(p, s, g, hyper_params.wn_decay)
                        for p, s, g in zip(p_leaves, s_leaves, g_leaves)))
    d_p, d_s, d_g, s_p, s_s, s_g = [
        jax.tree_unflatten(treedef, x) for x in split_grads]
    wn_params = {'direction': d_p, 'scale': s_p}
    wn_state = {'direction': d_s, 'scale': s_s}
    wn_grads = {'direction': d_g, 'scale': s_g}
    new_wn_params, new_state = self.wrapped_optimizer.apply_gradient(
        hyper_params.inner, wn_params,
        state.replace(param_states=wn_state), wn_grads)

    directions = treedef.flatten_up_to(new_wn_params['direction'])
    scales = treedef.flatten_up_to(new_wn_params['scale'])
    new_params, mults = zip(*(self._merge_param(d, s, hyper_params.wn_eps)
                              for d, s in zip(directions, scales)))
    new_params = jax.tree_unflatten(treedef, new_params)
    mults = jax.tree_unflatten(treedef, mults)

    direction_state = new_state.param_states['direction']
    scale_state = new_state.param_states['scale']
    param_states = jax.tree_multimap(
        lambda _, *args: _WeightNormParamState(*args),
        params, direction_state, scale_state, mults)
    return new_params, new_state.replace(param_states=param_states)

  def _split_param(self, param):
    if param.size > param.shape[-1]:
      scale = jnp.sqrt(jnp.square(param).sum(
          tuple(range(param.ndim-1)), keepdims=True))
      direction = param / scale
      return direction, scale
    else:
      return param, ()

  def _merge_param(self, direction, scale, eps):
    if direction.size > direction.shape[-1]:
      norm = jnp.sqrt(jnp.square(direction).sum(
          tuple(range(direction.ndim - 1)), keepdims=True))
      mult = scale / (eps + norm)
      param = direction * mult
      return param, mult
    else:
      return direction, ()

  def _split_grad(self, param, state, grad, decay):
    """Split the gradient for the direction and scale."""
    if param.size > param.shape[-1]:
      red_dims = tuple(range(param.ndim-1))
      direction = param / state.mult
      norm = jnp.sqrt(jnp.square(param).sum(red_dims, keepdims=True))
      scale = norm * jnp.sign(state.mult)
      scale_grad = jnp.sum(
          grad * direction, axis=red_dims, keepdims=True)
      direction_grad = state.mult * (grad - scale_grad * direction)
      if decay is not 0:
        direction_grad = direction_grad + decay * direction
      direction_info = direction, state.direction_state, direction_grad
      scale_info = scale, state.scale_state, scale_grad
      return direction_info + scale_info
    else:
      return (param, state.direction_state, grad, (), (), ())
