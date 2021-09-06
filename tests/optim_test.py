# Copyright 2021 The Flax Authors.
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

"""Tests for flax.optim."""

from functools import partial
from absl.testing import absltest
from flax import nn
from flax import optim
from flax import traverse_util
from flax.core.frozen_dict import FrozenDict
from flax.optim.adadelta import _AdadeltaHyperParams, _AdadeltaParamState
from flax.optim.adafactor import _AdafactorHyperParams, _AdafactorParamState
from flax.optim.adagrad import _AdagradHyperParams, _AdagradParamState
from flax.optim.adam import _AdamHyperParams, _AdamParamState
from flax.optim.adabelief import _AdaBeliefHyperParams, _AdaBeliefParamState
from flax.optim.momentum import _MomentumHyperParams, _MomentumParamState
from flax.optim.rmsprop import _RMSPropHyperParams, _RMSPropParamState
from flax.optim.sgd import _GradientDescentHyperParams
from flax.optim.weight_norm import _WeightNormParamState
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


def _assert_numpy_allclose(a, b, atol=None, rtol=None):
  a, b = jnp.array(a), jnp.array(b)
  a = a.astype(np.float32) if a.dtype == jnp.bfloat16 else a
  b = b.astype(np.float32) if b.dtype == jnp.bfloat16 else b
  kw = {}
  if atol: kw["atol"] = atol
  if rtol: kw["rtol"] = rtol
  np.testing.assert_allclose(a, b, **kw)


def check_eq(xs, ys, atol=None, rtol=None):
  xs_leaves, xs_tree = jax.tree_flatten(xs)
  ys_leaves, ys_tree = jax.tree_flatten(ys)
  assert xs_tree == ys_tree, "Tree shapes don't match."
  assert jax.tree_util.tree_all(jax.tree_multimap(
      lambda x, y: np.array(x).shape == np.array(y).shape,
      xs_leaves, ys_leaves)), "Leaves' shapes don't match."
  assert jax.tree_multimap(
      partial(_assert_numpy_allclose, atol=atol, rtol=rtol),
      xs_leaves, ys_leaves)


class OptimizerDefTest(absltest.TestCase):

  def test_create(self):
    params = np.ones((1,))
    optimizer_def = optim.Momentum(learning_rate=0.1, beta=0.2)
    optimizer = optimizer_def.create(params)
    expected_state = optim.OptimizerState(
        0, _MomentumParamState(np.zeros((1,))))
    self.assertEqual(optimizer.optimizer_def, optimizer_def)
    self.assertEqual(optimizer.state, expected_state)
    self.assertEqual(optimizer.target, params)

  @pytest.mark.filterwarnings("ignore: compute_gradient()")
  def test_compute_grad(self):
    params = np.ones(())
    optimizer_def = optim.Momentum(learning_rate=0.1, beta=0.2)
    optimizer = optimizer_def.create(params)
    def loss_fn(x):
      return 2. * x
    loss, grad = optimizer.compute_gradient(loss_fn)
    self.assertEqual(loss, 2.)
    self.assertEqual(grad, 2.)

    def loss_aux_fn(x):
      return 3. * x, 4.
    loss, aux, grad = optimizer.compute_gradient(loss_aux_fn)
    self.assertEqual(loss, 3.)
    self.assertEqual(grad, 3.)
    self.assertEqual(aux, 4.)

  def test_optimizer_with_focus(self):
    params = {'a': 0., 'b': 0.}
    opt_def = optim.GradientDescent(learning_rate=1.)
    t_a = traverse_util.t_identity['a']
    optimizer = opt_def.create(params, focus=t_a)
    expected_state = optim.OptimizerState(0, {'a': (), 'b': None})
    self.assertEqual(optimizer.state, expected_state)
    grads = {'a': -1., 'b': -2.}
    new_optimizer = optimizer.apply_gradient(grads)
    expected_params = {'a': 1., 'b': 0.}
    expected_state = optim.OptimizerState(1, {'a': (), 'b': None})
    self.assertEqual(new_optimizer.state, expected_state)
    self.assertEqual(new_optimizer.target, expected_params)

  def test_empty_optimizer(self):
    params = {}
    optimizer_def = optim.Momentum(learning_rate=0.1)
    optimizer = optimizer_def.create(params)
    new_optimizer = optimizer.apply_gradient({})
    expected_state = optim.OptimizerState(1, {})
    self.assertEqual(new_optimizer.state, expected_state)


class ModelParamTraversalTest(absltest.TestCase):

  def test_only_works_on_model_params(self):
    traversal = optim.ModelParamTraversal(lambda *_: True)
    with self.assertRaises(ValueError):
      list(traversal.iterate([]))

  def test_param_selection(self):
    params = {
        'x': {
            'kernel': 1,
            'bias': 2,
            'y': {
                'kernel': 3,
                'bias': 4,
            },
            'z': {},
        },
    }
    expected_params = {
        'x': {
            'kernel': 2,
            'bias': 2,
            'y': {
                'kernel': 6,
                'bias': 4,
            },
            'z': {}
        },
    }
    names = []
    def filter_fn(name, _):
      names.append(name)  # track names passed to filter_fn for testing
      return 'kernel' in name
    traversal = optim.ModelParamTraversal(filter_fn)

    # Model
    model = nn.Model(None, params)
    values = list(traversal.iterate(model))
    configs = [
      (nn.Model(None, params), nn.Model(None, expected_params)),
      (params, expected_params),
      (FrozenDict(params), FrozenDict(expected_params)),
    ]
    for model, expected_model in configs:
      self.assertEqual(values, [1, 3])
      self.assertEqual(set(names), set([
          '/x/kernel', '/x/bias', '/x/y/kernel', '/x/y/bias']))
      new_model = traversal.update(lambda x: x + x, model)
      self.assertEqual(new_model, expected_model)


class MultiOptimizerTest(absltest.TestCase):

  def test_multi_optimizer(self):
    params = {'a': 0., 'b': 0., 'c': {}}
    opt_a = optim.GradientDescent(learning_rate=1.)
    opt_b = optim.GradientDescent(learning_rate=10.)
    t_a = traverse_util.t_identity['a']
    t_b = traverse_util.t_identity['b']
    optimizer_def = optim.MultiOptimizer((t_a, opt_a), (t_b, opt_b))
    state = optimizer_def.init_state(params)
    expected_hyper_params = [
        _GradientDescentHyperParams(1.),
        _GradientDescentHyperParams(10.)
    ]
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = optim.OptimizerState(0, {'a': (), 'b': (), 'c': {}})
    self.assertEqual(state, expected_state)
    grads = {'a': -1., 'b': -2., 'c': {}}
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_params = {'a': 1., 'b': 20., 'c': {}}
    expected_state = optim.OptimizerState(1, {'a': (), 'b': (), 'c': {}})
    self.assertEqual(new_state, expected_state)
    self.assertEqual(new_params, expected_params)
    # override learning_rate
    hp = optimizer_def.update_hyper_params(learning_rate=2.)
    new_params, new_state = optimizer_def.apply_gradient(
        hp, params, state, grads)
    expected_params = {'a': 2., 'b': 4., 'c': {}}
    self.assertEqual(new_params, expected_params)

  def test_multi_optimizer_multiple_matches(self):
    params = {'a': {'x': 0., 'y': 0.}, 'b': {'y': 0, 'z': 0.}}
    opt_a = optim.GradientDescent(learning_rate=1.)
    opt_b = optim.GradientDescent(learning_rate=10.)
    t_a = optim.ModelParamTraversal(
      lambda path, _: path.endswith('/x') or path.endswith('/y')
    )
    t_b = optim.ModelParamTraversal(
      lambda path, value: value.dtype == jnp.int32 or path.endswith('/z')
    )
    optimizer_def = optim.MultiOptimizer((t_a, opt_a), (t_b, opt_b))
    with self.assertRaisesRegex(
        ValueError, r"Multiple optimizers match.*'y': \[0, 1\]"):
      jax.jit(optimizer_def.init_state)(params)


class GradientDescentTest(absltest.TestCase):

  def test_init_state(self):
    params = np.zeros((1,))
    optimizer_def = optim.GradientDescent(learning_rate=0.1)
    state = optimizer_def.init_state(params)
    expected_hyper_params = _GradientDescentHyperParams(0.1)
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = optim.OptimizerState(0, ())
    self.assertEqual(state, expected_state)

  def test_apply_gradient(self):
    optimizer_def = optim.GradientDescent(learning_rate=0.1)
    params = np.ones((1,))
    state = optim.OptimizerState(0, ())
    grads = np.array([3.])
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_new_state = optim.OptimizerState(1, ())
    expected_new_params = np.array([0.7])
    self.assertEqual(new_params, expected_new_params)
    self.assertEqual(new_state, expected_new_state)


class MomentumTest(absltest.TestCase):

  def test_init_state(self):
    params = np.zeros((1,))
    optimizer_def = optim.Momentum(learning_rate=0.1, beta=0.2)
    state = optimizer_def.init_state(params)
    expected_hyper_params = _MomentumHyperParams(0.1, 0.2, 0, False)
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = optim.OptimizerState(
        0, _MomentumParamState(np.zeros((1,))))
    self.assertEqual(state, expected_state)

  def test_apply_gradient(self):
    optimizer_def = optim.Momentum(learning_rate=0.1, beta=0.2)
    params = np.ones((1,))
    state = optim.OptimizerState(
        0, _MomentumParamState(np.array([1.])))
    grads = np.array([3.])
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_new_state = optim.OptimizerState(
        1, _MomentumParamState(np.array([3.2])))
    expected_new_params = np.array([1. - 0.32])
    self.assertEqual(new_params, expected_new_params)
    self.assertEqual(new_state, expected_new_state)


class AdamTest(absltest.TestCase):

  def test_init_state(self):
    params = np.zeros((1,))
    optimizer_def = optim.Adam(learning_rate=0.1,
                               beta1=0.2,
                               beta2=0.9,
                               eps=0.01,
                               weight_decay=0.0)
    state = optimizer_def.init_state(params)

    expected_hyper_params = _AdamHyperParams(0.1, 0.2, 0.9, 0.01, 0.0)
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = optim.OptimizerState(
        0, _AdamParamState(np.zeros((1,)), np.zeros((1,))))
    self.assertEqual(state, expected_state)

  def test_apply_gradient(self):
    optimizer_def = optim.Adam(learning_rate=0.1,
                               beta1=0.2,
                               beta2=0.9,
                               eps=0.01,
                               weight_decay=0.0)
    params = np.array([1.])
    state = optim.OptimizerState(
        1, _AdamParamState(np.array([0.1]), np.array([0.9])))
    grads = np.array([4.])
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_new_state = optim.OptimizerState(
        2, _AdamParamState(np.array([3.22]), np.array([2.41])))
    expected_new_params = np.array([0.906085])
    np.testing.assert_allclose(new_params, expected_new_params)
    self.assertEqual(new_state, expected_new_state)


class AdaBeliefTest(absltest.TestCase):

  def test_init_state(self):
    params = np.zeros((1,))
    optimizer_def = optim.AdaBelief(
        learning_rate=0.1, beta1=0.2, beta2=0.9, eps=0.01, weight_decay=0.0)
    state = optimizer_def.init_state(params)

    expected_hyper_params = _AdaBeliefHyperParams(0.1, 0.2, 0.9, 0.01, 0.0)
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = optim.OptimizerState(
        0, _AdaBeliefParamState(np.zeros((1,)), np.zeros((1,))))
    self.assertEqual(state, expected_state)

  def test_apply_gradient(self):
    optimizer_def = optim.AdaBelief(
        learning_rate=0.1, beta1=0.2, beta2=0.9, eps=0.01, weight_decay=0.0)
    params = np.array([1.])
    state = optim.OptimizerState(
        1, _AdaBeliefParamState(np.array([0.1]), np.array([0.9])))
    grads = np.array([4.])
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_new_state = optim.OptimizerState(
        2, _AdaBeliefParamState(np.array([3.22]), np.array([0.88084])))
    expected_new_params = np.array([0.8449397])
    np.testing.assert_allclose(new_params, expected_new_params)
    self.assertEqual(new_state, expected_new_state)


class AdadeltaTest(absltest.TestCase):

  def test_init_state(self):
    params = np.zeros((1,))
    optimizer_def = optim.Adadelta(learning_rate=0.1,
                                   rho=0.9,
                                   eps=1e-6,
                                   weight_decay=0.1)
    state = optimizer_def.init_state(params)

    expected_hyper_params = _AdadeltaHyperParams(0.1, 0.9, 1e-6, 0.1)
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = optim.OptimizerState(
        0, _AdadeltaParamState(np.zeros((1,)), np.zeros((1,)))
    )
    self.assertEqual(state, expected_state)

  def test_apply_gradient(self):
    optimizer_def = optim.Adadelta(learning_rate=0.1,
                                   rho=0.9,
                                   eps=1e-6,
                                   weight_decay=0.1)
    params = np.array([1.])
    state = optim.OptimizerState(
        1, _AdadeltaParamState(np.zeros((1,)), np.zeros((1,)))
    )
    grads = np.array([1.])
    new_param, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads
    )
    expected_new_state = optim.OptimizerState(
        2, _AdadeltaParamState(np.array([0.1]), np.array([9.999902e-7]))
    )
    expected_new_params = np.array([0.9896838])
    np.testing.assert_allclose(new_param, expected_new_params)
    self.assertEqual(new_state, expected_new_state)


class AdafactorTest(absltest.TestCase):

  def test_init_state(self):
    params = np.zeros((3, 2))
    optimizer_def = optim.Adafactor(learning_rate=0.1,
                                    decay_rate=0.8,
                                    beta1=None,
                                    min_dim_size_to_factor=0)
    state = optimizer_def.init_state(params)

    expected_hyper_params = _AdafactorHyperParams(0.1, True, True,
                                                  None, 0.8, 0, 1.0, None, 0,
                                                  1e-30, 1e-3)
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = optim.OptimizerState(
        0, _AdafactorParamState(np.zeros((2,)), np.zeros((3,)),
                                np.zeros((1,)), np.zeros((1,))))
    check_eq(state, expected_state)

    # unfactorized
    optimizer_def = optim.Adafactor(learning_rate=0.1,
                                    decay_rate=0.8,
                                    beta1=0.0,
                                    min_dim_size_to_factor=32)
    state = optimizer_def.init_state(params)

    expected_hyper_params = _AdafactorHyperParams(0.1, True, True,
                                                  0.0, 0.8, 0, 1.0, None, 32,
                                                  1e-30, 1e-3)
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = optim.OptimizerState(
        0, _AdafactorParamState(np.zeros((1,)), np.zeros((1,)),
                                np.zeros((3, 2)), np.zeros((3, 2))))
    check_eq(state, expected_state)

  def test_apply_gradient(self):
    optimizer_def = optim.Adafactor(learning_rate=0.1, decay_rate=0.8,
                                    min_dim_size_to_factor=0)
    params = np.ones((3, 2), np.float32)
    state = optim.OptimizerState(
        1, _AdafactorParamState(np.array([0.9, 0.9]),
                                np.array([0.1, 0.1, 0.1]),
                                np.zeros((1,)),
                                np.zeros((1,))))
    grads = np.ones((3, 2), np.float32)
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_new_state = optim.OptimizerState(
        2, _AdafactorParamState(
            np.array([0.9574349, 0.9574349]),
            np.array([0.6169143, 0.6169143, 0.6169143]),
            np.zeros((1,)),
            np.zeros((1,))))
    expected_new_params = 0.9 * np.ones((3, 2))
    np.testing.assert_allclose(new_params, expected_new_params)
    check_eq(new_state, expected_new_state, rtol=1e-6)

    # unfactored w momentum
    optimizer_def = optim.Adafactor(learning_rate=0.1,
                                    beta1=0.0, decay_rate=0.8,
                                    min_dim_size_to_factor=32)
    params = np.ones((3, 2), np.float32)
    state = optim.OptimizerState(
        1, _AdafactorParamState(np.zeros(1,),
                                np.zeros(1,),
                                0.5*np.ones((3, 2)),
                                np.zeros((3, 2))))
    grads = np.ones((3, 2), np.float32)
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_new_params = 0.9 * np.ones((3, 2))
    np.testing.assert_allclose(new_params, expected_new_params)
    expected_new_state = optim.OptimizerState(
        2, _AdafactorParamState(
            np.array([0.0]),
            np.array([0.0]),
            0.787174 * np.ones((3, 2)),
            0.1 * np.ones((3,2))))
    check_eq(new_state, expected_new_state, rtol=1e-6)

  def test_factorizes(self):
    params = np.zeros((64, 64))
    optimizer_def = optim.Adafactor(learning_rate=0.1,
                                    decay_rate=0.8,
                                    beta1=None,
                                    min_dim_size_to_factor=32)
    state = optimizer_def.init_state(params)
    self.assertEqual(state.param_states.v.shape, (1,))
    self.assertEqual(state.param_states.m.shape, (1,))
    self.assertEqual(state.param_states.v_row.shape, (64,))
    self.assertEqual(state.param_states.v_col.shape, (64,))

    params = np.zeros((31, 64))
    optimizer_def = optim.Adafactor(learning_rate=0.1,
                                    decay_rate=0.8,
                                    beta1=None,
                                    min_dim_size_to_factor=32)
    state = optimizer_def.init_state(params)
    self.assertEqual(state.param_states.v.shape, (31, 64))
    self.assertEqual(state.param_states.m.shape, (1,))
    self.assertEqual(state.param_states.v_row.shape, (1,))
    self.assertEqual(state.param_states.v_col.shape, (1,))


class AdagradTest(absltest.TestCase):

  def test_init_state(self):
    params = np.zeros((1,))
    optimizer_def = optim.Adagrad(learning_rate=0.1, eps=0.01)
    state = optimizer_def.init_state(params)

    expected_hyper_params = _AdagradHyperParams(0.1, 0.01)
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = optim.OptimizerState(
        0, _AdagradParamState(np.zeros((1,))))
    self.assertEqual(state, expected_state)

  def test_apply_gradient(self):
    optimizer_def = optim.Adagrad(learning_rate=0.1, eps=0.01)
    params = np.array([1.])
    state = optim.OptimizerState(
        1, _AdagradParamState(np.array([0.1])))
    grads = np.array([4.])
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_new_state = optim.OptimizerState(
        2, _AdagradParamState(np.array([16.1])))
    expected_new_params = np.array([0.9005588])
    np.testing.assert_allclose(new_params, expected_new_params)
    self.assertEqual(new_state, expected_new_state)


class RMSPropTest(absltest.TestCase):

  def test_init_state(self):
    params = np.zeros((1,))
    optimizer_def = optim.RMSProp(learning_rate=0.1,
                                  beta2=0.9,
                                  eps=0.01,
                                  centered=False)
    state = optimizer_def.init_state(params)

    expected_hyper_params = _RMSPropHyperParams(0.1, 0.9, 0.01, False)
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = optim.OptimizerState(
        0, _RMSPropParamState(np.zeros((1,)), None))
    self.assertEqual(state, expected_state)

  def test_init_state_centered(self):
    params = np.zeros((1,))
    optimizer_def = optim.RMSProp(learning_rate=0.1,
                                  beta2=0.9,
                                  eps=0.01,
                                  centered=True)
    state = optimizer_def.init_state(params)

    expected_hyper_params = _RMSPropHyperParams(0.1, 0.9, 0.01, True)
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = optim.OptimizerState(
        0, _RMSPropParamState(np.zeros((1,)), np.zeros((1,))))
    self.assertEqual(state, expected_state)

  def test_apply_gradient(self):
    optimizer_def = optim.RMSProp(learning_rate=0.1,
                                  beta2=0.9,
                                  eps=0.01)
    params = np.array([1.])
    state = optim.OptimizerState(
        1, _RMSPropParamState(np.array([0.1]), None))
    grads = np.array([4.])
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_new_state = optim.OptimizerState(
        2, _RMSPropParamState(np.array([1.69]), None))
    expected_new_params = np.array([0.6946565])
    np.testing.assert_allclose(new_params, expected_new_params)
    self.assertEqual(new_state, expected_new_state)

  def test_apply_gradient_centered(self):
    optimizer_def = optim.RMSProp(learning_rate=0.1,
                                  beta2=0.9,
                                  eps=0.01,
                                  centered=True)
    params = np.array([1.])
    state = optim.OptimizerState(
        1, _RMSPropParamState(np.array([0.1]), np.array([0.1])))
    grads = np.array([4.])
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_new_state = optim.OptimizerState(
        2, _RMSPropParamState(np.array([1.69]), np.array([0.49])))
    expected_new_params = np.array([0.670543], dtype=np.float32)
    np.testing.assert_allclose(new_params, expected_new_params, rtol=1e-6)
    np.testing.assert_allclose(new_state.param_states.v,
                                expected_new_state.param_states.v)
    np.testing.assert_allclose(new_state.param_states.mg,
                                expected_new_state.param_states.mg)


class WeightNormTest(absltest.TestCase):

  def test_momentum_with_weight_norm(self):
    params = np.ones((2, 2)) * 2.
    optimizer_def = optim.WeightNorm(optim.Momentum(0.1))
    state = optimizer_def.init_state(params)
    self.assertEqual(jax.tree_map(np.shape, state), optim.OptimizerState(
        step=(),
        param_states=_WeightNormParamState(
            direction_state=_MomentumParamState(momentum=(2, 2)),
            scale_state=_MomentumParamState(momentum=(1, 2)),
            mult=(1, 2)
        )
    ))
    grads = np.ones((2, 2))
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    np.testing.assert_allclose(new_params, np.full_like(params, 1.9))
    np.testing.assert_allclose(new_state.param_states.mult, 1.9 * 2 ** 0.5)


class DynamicScaleTest(absltest.TestCase):

  def test_dynamic_scale(self):
    def loss_fn(p):
      return jnp.asarray(p, jnp.float16) ** 2
    p = jnp.array(1., jnp.float32)

    dyn_scale = optim.DynamicScale(growth_interval=2)
    step = jax.jit(lambda ds, p: ds.value_and_grad(loss_fn)(p))
    inf = float('inf')
    nan = float('nan')
    expected_values = [
        (False, nan, 32768.0, inf),
        (False, 1.0, 16384.0, inf),
        (True, 1.0, 16384.0, 2.0),
        (True, 1.0, 16384.0, 2.0),
        (True, 1.0, 32768.0, 2.0),
        (False, 1.0, 16384.0, inf),
    ]

    for expected in expected_values:
      dyn_scale, is_fin, loss, grad = step(dyn_scale, p)
      values = np.array((is_fin, loss, dyn_scale.scale, grad))
      np.testing.assert_allclose(values, expected)

if __name__ == '__main__':
  absltest.main()
