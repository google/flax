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

"""Tests for flax.nn."""

import threading
from absl.testing import absltest

from flax import nn

import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp

import numpy as onp

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class DummyModule(nn.Module):

  def apply(self, x):
    bias = self.param('bias', x.shape, initializers.ones)
    return x + bias


class NestedModule(nn.Module):

  def apply(self, x):
    x = DummyModule(x, name='dummy_0')
    x = DummyModule(x, name='dummy_1')
    return x


class NestedModel(nn.Module):

  def apply(self, x, model):
    x = DummyModule(x, name='dummy_0')
    x = model(x, name='inner_model')
    return x


class DataDependentInitModule(nn.Module):

  def apply(self, x):
    bias = self.param('bias', x.shape, lambda rng, shape: x + 1.)
    return x + bias


class CollectionModule(nn.Module):

  def apply(self, x, activations=None):
    bias = self.param('bias', x.shape, initializers.ones)
    y = x + bias
    if activations:
      previous_activation = activations.retrieve()
      activations.store(y)
      return y, previous_activation
    else:
      return y, None


class LoopModule(nn.Module):

  def apply(self, x, activations=None):
    module = CollectionModule.shared(activations=activations, name='dummy')
    for _ in range(2):
      x, _ = module(x)
    return x


class ModuleTest(absltest.TestCase):

  def test_init_module(self):
    rng = random.PRNGKey(0)
    x = jnp.array([1.])
    y, params = DummyModule.init(rng, x)
    y2 = DummyModule.call(params, x)
    self.assertEqual(y, y2)
    self.assertEqual(y, jnp.array([2.]))
    self.assertEqual(params, {'bias': jnp.array([1.])})

  def test_init_by_shape_module(self):
    rng = random.PRNGKey(0)
    x = jnp.array([1.])
    y, params = DummyModule.init_by_shape(rng, [(x.shape, x.dtype)])
    y2 = DummyModule.call(params, x)
    self.assertEqual(y.shape, y2.shape)
    self.assertEqual(y2, jnp.array([2.]))
    self.assertEqual(params, {'bias': jnp.array([1.])})

  def test_shared_module(self):
    rng = random.PRNGKey(0)
    x = jnp.array([1.])
    _, initial_params = LoopModule.init(rng, x)
    model = nn.Model(LoopModule, initial_params)
    y = model(x)
    self.assertEqual(y, jnp.array([3.]))
    self.assertEqual(model.params, {'dummy': {'bias': jnp.array([1.])}})

  def test_name_collsion(self):
    class FaultyModule(nn.Module):

      def apply(self, x):
        for _ in range(2):
          DummyModule(x, name='dummy')

    x = jnp.array([1.])
    with self.assertRaises(ValueError):
      FaultyModule.init(random.PRNGKey(0), x)

  def test_sharing_name_collsion(self):
    class FaultyModule(nn.Module):

      def apply(self, x):
        for _ in range(2):
          module = DummyModule.shared(name='dummy')
          module(x)

    x = jnp.array([1.])
    with self.assertRaises(ValueError):
      FaultyModule.init(random.PRNGKey(0), x)

  def test_sharing_name_on_apply(self):
    class FaultyModule(nn.Module):

      def apply(self, x):
        module = DummyModule.shared(name='dummy')
        for _ in range(2):
          module(x, name='dummy2')

    x = jnp.array([1.])
    with self.assertRaises(ValueError):
      FaultyModule.init(random.PRNGKey(0), x)

  def test_shared_module_called_in_other_frame(self):
    """Test that shared modules only appear once in parameters.

    Concretely, create a shared submodule, then pass it in to
    a child module and apply it there. Test that the parameters
    are only stored once, in the frame where the shared module
    was created.
    """

    class SubModule(nn.Module):

      def apply(self):
        self.param('param', (), initializers.zeros)

    class UseSharedModule(nn.Module):

      def apply(self, submodule):
        submodule()

    class TopLevel(nn.Module):

      def apply(self):
        submodule = SubModule.shared(name='shared')
        submodule()
        UseSharedModule(submodule, name='use_shared')

    _, params = TopLevel.init(random.PRNGKey(0))
    self.assertEqual({
        'shared': {'param': jnp.zeros(())},
        'use_shared': {},
    }, params)

  def test_module_decorator(self):
    @nn.module
    def MyModule(x):  # pylint: disable=invalid-name
      return DummyModule(x)

    self.assertEqual(MyModule.__name__, 'MyModule')
    self.assertTrue(issubclass(MyModule, nn.Module))

    rng = random.PRNGKey(0)
    x = jnp.array([1.])
    y, params = MyModule.init(rng, x)
    y2 = MyModule.call(params, x)
    self.assertEqual(y, y2)
    self.assertEqual(y, jnp.array([2.]))

  def test_partial_application(self):
    rng = random.PRNGKey(0)
    x = jnp.array([1.])
    dummy_module = DummyModule.partial(x=x)  # partially apply the inputs
    y, initial_params = dummy_module.init(rng)
    model = nn.Model(dummy_module, initial_params)
    y2 = model()
    self.assertEqual(y.shape, y2.shape)
    self.assertEqual(y2, jnp.array([2.]))

  def test_nested_model(self):
    x = jnp.array([1.])
    _, inner_initial_params = DummyModule.init(random.PRNGKey(0), x)
    inner_model = nn.Model(DummyModule, inner_initial_params)
    _, initial_params = NestedModel.init(random.PRNGKey(1), x, inner_model)
    model = nn.Model(NestedModel, initial_params)
    y = model(x, inner_model)
    self.assertEqual(y, jnp.array([3.]))

  def test_capture_module_outputs(self):
    x = jnp.array([1.])
    _, initial_params = NestedModule.init(random.PRNGKey(0), x)
    model = nn.Model(NestedModule, initial_params)
    with nn.capture_module_outputs() as activations:
      model(x)
    expected_activations = {
        '/': [x + 2],
        '/dummy_0': [x + 1],
        '/dummy_1': [x + 2],
    }
    self.assertEqual(activations.as_dict(), expected_activations)

  def test_nested_model_capture_outputs(self):
    x = jnp.array([1.])
    _, inner_initial_params = DummyModule.init(random.PRNGKey(0), x)
    inner_model = nn.Model(DummyModule, inner_initial_params)
    _, initial_params = NestedModel.init(random.PRNGKey(1), x, inner_model)
    model = nn.Model(NestedModel, initial_params)
    with nn.capture_module_outputs() as activations:
      model(x, inner_model)
    expected_activations = {
        '/': [x + 2],
        '/dummy_0': [x + 1],
        '/inner_model': [x + 2],
    }
    self.assertEqual(activations.as_dict(), expected_activations)

  def test_truncated_module(self):
    x = jnp.array([1.])
    _, initial_params = NestedModule.init(random.PRNGKey(0), x)
    model = nn.Model(NestedModule, initial_params)
    model = model.truncate_at('/dummy_0')
    y = model(x)
    self.assertEqual(y, [x + 1])

  def test_call_module_method(self):
    class MultiMethod(nn.Module):

      def apply(self, x):
        return x + self.param('bias', x.shape, initializers.ones)

      @nn.module_method
      def l2(self):
        return jnp.sum(self.get_param('bias') ** 2)

    class MultiMethodModel(nn.Module):

      def apply(self, x):
        layer = MultiMethod.shared()
        layer(x)  # init
        return layer.l2()

    x = jnp.array([1., 2.])
    y, _ = MultiMethodModel.init(random.PRNGKey(0), x)
    self.assertEqual(y, 2.)

  def test_module_state(self):
    class StatefulModule(nn.Module):

      def apply(self, x, coll=None):
        state = self.state('state', x.shape, nn.initializers.zeros,
                           collection=coll)
        state.value += x

    x = jnp.array([1.,])
    # no collection should raise an error
    with self.assertRaises(ValueError):
      StatefulModule.call({}, x)

    # pass collection explicitly
    with nn.Collection().mutate() as state:
      self.assertEqual(state.as_dict(), {})
      StatefulModule.init(random.PRNGKey(0), x, state)
      self.assertEqual(state.as_dict(), {'/': {'state': x}})
    self.assertEqual(state.as_dict(), {'/': {'state': x}})
    with state.mutate() as new_state:
      # assert new_state is a clone of state
      self.assertEqual(new_state.as_dict(), state.as_dict())
      StatefulModule.call({}, x, new_state)
    self.assertEqual(new_state.as_dict(), {'/': {'state': x + x}})

    # use stateful
    with nn.stateful() as state:
      self.assertEqual(state.as_dict(), {})
      StatefulModule.init(random.PRNGKey(0), x)
    self.assertEqual(state.as_dict(), {'/': {'state': x}})
    with nn.stateful(state) as new_state:
      # assert new_state is a clone of state
      self.assertEqual(new_state.as_dict(), state.as_dict())
      StatefulModule.call({}, x)
      self.assertEqual(new_state.as_dict(), {'/': {'state': x + x}})
    self.assertEqual(new_state.as_dict(), {'/': {'state': x + x}})

  def test_parameter_rng(self):
    @nn.module
    def model(x):
      return nn.Dense(x, features=2, name='dummy',
                      bias_init=nn.initializers.normal())
    rng = random.PRNGKey(0)
    _, params = model.init(rng, jnp.ones((1, 1)))
    dense_rng = nn.base._fold_in_str(rng, 'dummy')
    kernel_rng = nn.base._fold_in_str(dense_rng, 'kernel')
    bias_rng = nn.base._fold_in_str(dense_rng, 'bias')
    kernel = nn.linear.default_kernel_init(kernel_rng, (1, 2))
    bias = nn.initializers.normal()(bias_rng, (2,))
    onp.testing.assert_allclose(kernel, params['dummy']['kernel'])
    onp.testing.assert_allclose(bias, params['dummy']['bias'])

class CollectionTest(absltest.TestCase):

  def test_collection_store_and_retrieve(self):
    rng = random.PRNGKey(0)
    x = jnp.array([1.])
    with nn.Collection().mutate() as activations:
      (_, y), initial_params = CollectionModule.init(rng, x, activations)
      model = nn.Model(CollectionModule, initial_params)
      self.assertEqual(y, None)
    with activations.mutate() as new_activations:
      _, y2 = model(x, new_activations)
    self.assertEqual(y2, jnp.array([2.]))

  def test_collection_multiple_calls(self):
    rng = random.PRNGKey(0)
    with nn.Collection().mutate() as activations:
      x = jnp.array([1.])
      _, _ = LoopModule.init(rng, x, activations)
    expected_state = {
        '/dummy': jnp.array([3.]),
    }
    self.assertEqual(activations.state, expected_state)

  def test_collection_multiple_roots(self):
    rng = random.PRNGKey(0)
    with nn.Collection().mutate() as activations:
      x = jnp.array([1.])
      LoopModule.init(rng, x, activations, name='a')
      LoopModule.init(rng, x, activations, name='b')
    expected_state = {
        '/a/dummy': jnp.array([3.]),
        '/b/dummy': jnp.array([3.]),
    }
    self.assertEqual(activations.state, expected_state)
    with self.assertRaises(ValueError):
      with nn.Collection().mutate() as activations:
        x = jnp.array([1.])
        LoopModule.init(rng, x, activations)
        LoopModule.init(rng, x, activations)

  def test_mutable_collection_cannot_be_passed_to_jax(self):
    with nn.Collection().mutate() as collection:
      def fn(col):
        return col
      with self.assertRaises(ValueError):
        jax.jit(fn)(collection)

  def test_collection_lookup(self):
    state = {
        '/dummy/sub': 1,
    }
    collection = nn.Collection(state=state)
    root = nn.base._ModuleFrame(None)
    frame = nn.base._ModuleFrame('dummy', parent=root)
    with nn.base._module_stack.frame(root):
      with nn.base._module_stack.frame(frame):
        self.assertEqual(collection['/dummy/sub'], 1)

  def test_collection_inside_module(self):
    class NestedCollection(nn.Module):

      def apply(self, x):
        with nn.Collection().mutate() as activations:
          LoopModule(x, activations, name='a')
          LoopModule(x, activations, name='b')
        return activations

    rng = random.PRNGKey(0)
    x = jnp.array([1.])
    activations, _ = NestedCollection.init(rng, x, name='nested')
    expected_state = {
        '/a/dummy': jnp.array([3.]),
        '/b/dummy': jnp.array([3.]),
    }
    self.assertEqual(activations.as_dict(), expected_state)

  def test_collection_store_fails_if_not_in_module(self):
    @nn.module
    def test():
      with nn.Collection().mutate() as coll:
        coll.store(1)
    pattern = 'State should be stored from within a module'
    with self.assertRaisesRegex(ValueError, pattern):
      test.init(random.PRNGKey(0))

  def test_collection_store_fails_if_out_of_scope(self):
    @nn.module
    def stateful_module(coll):
      coll.store(1)

    @nn.module
    def test_inner(f):
      with nn.Collection().mutate() as coll:
        # this should fail because f is a shared module defined
        # in the parent. Therefore we cannot capture in the scope
        # of this Module.
        f(coll)

    @nn.module
    def test():
      f = stateful_module.shared()
      test_inner(f)
    pattern = 'Trying to capture state outside the scope'
    with self.assertRaisesRegex(ValueError, pattern):
      test.init(random.PRNGKey(0))

  def test_jax_transform_of_stateful_function(self):
    test = self
    class NestedTransform(nn.Module):

      def apply(self, state, y):
        def inner_fn(x):
          # constants should be storable
          state.store(1.)
          # values in the same trace should be storable
          state.store(y)
          with test.assertRaises(ValueError):
            # values depending on the vmap should not be storable
            state.store(x)
        jax.vmap(inner_fn)(jnp.ones((2,)))

    def outer_fn(x):
      with nn.Collection().mutate() as state:
        NestedTransform.init(random.PRNGKey(0), state, x)

    outer_fn(1.)
    jax.jit(outer_fn)(1.)


class UtilsTest(absltest.TestCase):

  def test_call_stack_happy_path(self):
    stack = nn.utils.CallStack()
    self.assertFalse(stack)
    with stack.frame({'id': 1}):
      self.assertTrue(stack)
      self.assertEqual(stack[-1], {'id': 1})
      with stack.frame({'id': 2}):
        self.assertEqual(list(stack), [{'id': 1}, {'id': 2}])
      self.assertEqual(list(stack), [{'id': 1}])

  def test_call_stack_multithreading(self):
    stack = nn.utils.CallStack()
    self.assertFalse(stack)
    with stack.frame({'id': 1}):
      self.assertEqual(stack[-1], {'id': 1})
      def _main():
        # Each thread should have its own stack.
        self.assertFalse(stack)
        with stack.frame({'id': 2}):
          self.assertEqual(stack[-1], {'id': 2})
      thread = threading.Thread(target=_main)
      thread.start()
      thread.join()

  def test_call_stack_error_path(self):
    stack = nn.utils.CallStack()
    with stack.frame({'id': 1}):
      with self.assertRaises(ValueError):
        with stack.frame({'id': 2}):
          raise ValueError('dummy')
      self.assertEqual(list(stack), [{'id': 1}])


class PoolTest(absltest.TestCase):

  def test_pool_custom_reduce(self):
    x = jnp.full((1, 3, 3, 1), 2.)
    mul_reduce = lambda x, y: x * y
    y = nn.pooling.pool(x, 1., mul_reduce, (2, 2), (1, 1), 'VALID')
    onp.testing.assert_allclose(y, onp.full((1, 2, 2, 1), 2. ** 4))

  def test_avg_pool(self):
    x = jnp.full((1, 3, 3, 1), 2.)
    pool = lambda x: nn.avg_pool(x, (2, 2))
    y = pool(x)
    onp.testing.assert_allclose(y, onp.full((1, 2, 2, 1), 2.))
    y_grad = jax.grad(lambda x: pool(x).sum())(x)
    expected_grad = jnp.array([
        [0.25, 0.5, 0.25],
        [0.5, 1., 0.5],
        [0.25, 0.5, 0.25],
    ]).reshape((1, 3, 3, 1))
    onp.testing.assert_allclose(y_grad, expected_grad)

  def test_max_pool(self):
    x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)
    pool = lambda x: nn.max_pool(x, (2, 2))
    expected_y = jnp.array([
        [4., 5.],
        [7., 8.],
    ]).reshape((1, 2, 2, 1))
    y = pool(x)
    onp.testing.assert_allclose(y, expected_y)
    y_grad = jax.grad(lambda x: pool(x).sum())(x)
    expected_grad = jnp.array([
        [0., 0., 0.],
        [0., 1., 1.],
        [0., 1., 1.],
    ]).reshape((1, 3, 3, 1))
    onp.testing.assert_allclose(y_grad, expected_grad)


class NormalizationTest(absltest.TestCase):

  def test_batch_norm(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (4, 3, 2))
    model_cls = nn.BatchNorm.partial(momentum=0.9)
    with nn.stateful() as state_0:
      y, initial_params = model_cls.init(key2, x)
      model = nn.Model(model_cls, initial_params)
    mean = y.mean((0, 1))
    var = y.var((0, 1))
    onp.testing.assert_allclose(mean, onp.array([0., 0.]), atol=1e-4)
    onp.testing.assert_allclose(var, onp.array([1., 1.]), rtol=1e-4)
    with nn.stateful(state_0) as state:
      y = model(x)
    ema = state['/']
    onp.testing.assert_allclose(
        ema['mean'], 0.1 * x.mean((0, 1), keepdims=False), atol=1e-4)
    onp.testing.assert_allclose(
        ema['var'], 0.9 + 0.1 * x.var((0, 1), keepdims=False), rtol=1e-4)

  def test_layer_norm(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 3, 4))
    y, _ = nn.LayerNorm.init(key2, x, bias=False, scale=False, epsilon=e)
    assert x.shape == y.shape
    input_type = type(x)
    assert  isinstance(y, input_type)
    y_one_liner = ((x - x.mean(axis=-1, keepdims=True)) *
                   jax.lax.rsqrt(x.var(axis=-1, keepdims=True) + e))
    onp.testing.assert_allclose(y_one_liner, y, atol=1e-4)

  def test_group_norm(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 5, 4, 4, 32))
    y, _ = nn.GroupNorm.init(key2, x, num_groups=2,
                               bias=True, scale=True, epsilon=e)
    self.assertEqual(x.shape, y.shape)
    self.assertIsInstance(y, type(x))

    x_gr = x.reshape([2, 5, 4, 4, 2, 16])
    y_test = ((x_gr - x_gr.mean(axis=[1, 2, 3, 5], keepdims=True)) *
              jax.lax.rsqrt(x_gr.var(axis=[1, 2, 3, 5], keepdims=True) + e))
    y_test = y_test.reshape([2, 5, 4, 4, 32])

    onp.testing.assert_allclose(y_test, y, atol=1e-4)


# TODO(flax-dev): add integration tests for RNN cells
class RecurrentTest(absltest.TestCase):

  def test_lstm(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (2, 3))
    c0, h0 = nn.LSTMCell.initialize_carry(rng, (2,), 4)
    self.assertEqual(c0.shape, (2, 4))
    self.assertEqual(h0.shape, (2, 4))
    (carry, y), initial_params = nn.LSTMCell.init(key2, (c0, h0), x)
    lstm = nn.Model(nn.LSTMCell, initial_params)
    self.assertEqual(carry[0].shape, (2, 4))
    self.assertEqual(carry[1].shape, (2, 4))
    onp.testing.assert_allclose(y, carry[1])
    param_shapes = jax.tree_map(onp.shape, lstm.params)
    self.assertEqual(param_shapes, {
        'ii': {'kernel': (3, 4)},
        'if': {'kernel': (3, 4)},
        'ig': {'kernel': (3, 4)},
        'io': {'kernel': (3, 4)},
        'hi': {'kernel': (4, 4), 'bias': (4,)},
        'hf': {'kernel': (4, 4), 'bias': (4,)},
        'hg': {'kernel': (4, 4), 'bias': (4,)},
        'ho': {'kernel': (4, 4), 'bias': (4,)},
    })

  def test_gru(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (2, 3))
    carry0 = nn.GRUCell.initialize_carry(rng, (2,), 4)
    self.assertEqual(carry0.shape, (2, 4))
    (carry, y), initial_params = nn.GRUCell.init(key2, carry0, x)
    gru = nn.Model(nn.GRUCell, initial_params)
    self.assertEqual(carry.shape, (2, 4))
    onp.testing.assert_allclose(y, carry)
    param_shapes = jax.tree_map(onp.shape, gru.params)
    self.assertEqual(param_shapes, {
        'ir': {'kernel': (3, 4), 'bias': (4,)},
        'iz': {'kernel': (3, 4), 'bias': (4,)},
        'in': {'kernel': (3, 4), 'bias': (4,)},
        'hr': {'kernel': (4, 4)},
        'hz': {'kernel': (4, 4)},
        'hn': {'kernel': (4, 4), 'bias': (4,)},
    })


class StochasticTest(absltest.TestCase):

  def test_make_rng_requires_stochastic(self):
    with self.assertRaises(ValueError):
      nn.make_rng()

  def test_stochastic_rngs(self):
    rng = random.PRNGKey(0)
    with nn.stochastic(rng):
      r1 = nn.make_rng()
      r2 = nn.make_rng()
    self.assertTrue(onp.all(r1 == random.fold_in(rng, 1)))
    self.assertTrue(onp.all(r2 == random.fold_in(rng, 2)))

  def test_make_rng_in_jax_transform_check(self):
    with nn.stochastic(random.PRNGKey(0)):
      with self.assertRaises(ValueError):
        jax.jit(nn.make_rng)()

  def test_init_by_shape_lifts_stochastic(self):
    class StochasticModule(nn.Module):
      def apply(self):
        return nn.make_rng()

    with nn.stochastic(random.PRNGKey(0)):
      rng, _ = StochasticModule.init_by_shape(random.PRNGKey(1), [])
    expected_rng = random.fold_in(random.PRNGKey(0), 1)
    expected_rng = random.fold_in(expected_rng, 1)
    self.assertTrue(onp.all(rng == expected_rng))


if __name__ == '__main__':
  absltest.main()
