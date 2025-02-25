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

import os
from typing import Any

from flax.linen.dtypes import promote_dtype

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import flax
from flax import linen as nn
from flax import nnx
from flax.nnx import bridge
from flax.nnx.bridge.module import MODULE_CONTEXT


class TestCompatibility(absltest.TestCase):
  def setUp(self):
    super().setUp()
    dim1 = max(jax.device_count() // 2, 1)
    device_mesh = np.array(jax.devices()).reshape(dim1, jax.device_count() // dim1)
    self.mesh = jax.sharding.Mesh(devices=device_mesh, axis_names=('in', 'out'))

  def test_functional(self):
    # Functional API for NNX Modules
    functional = bridge.functional(nnx.Linear)(32, 64)
    state = functional.init(rngs=nnx.Rngs(0))
    x = jax.numpy.ones((1, 32))
    y, updates = functional.apply(state)(x)

  ##################
  ### LinenToNNX ###
  ##################

  def test_linen_to_nnx(self):
    ## Wrapper API for Linen Modules
    linen_module = nn.Dense(features=64)
    x = jax.numpy.ones((1, 32))
    model = bridge.ToNNX(linen_module, rngs=nnx.Rngs(0)).lazy_init(x)  # like linen init
    y = model(x)  # like linen apply
    assert y.shape == (1, 64)
    self.assertIsInstance(model.kernel, nnx.Variable)
    # NNX automatically adds metadata box regardless of original Linen module.
    linen_vars = linen_module.init(jax.random.key(0), x)
    np.testing.assert_array_equal(linen_vars['params']['kernel'],
                                  model.kernel.value)

  def test_linen_to_nnx_submodule(self):
    class NNXOuter(nnx.Module):
      def __init__(self, dout: int, *, rngs: nnx.Rngs):
        self.nn_dense1 = bridge.ToNNX(nn.Dense(dout, use_bias=False), rngs=rngs)
        self.b = nnx.Param(jax.random.uniform(rngs.params(), (1, dout,)))
        self.batchnorm = bridge.ToNNX(nn.BatchNorm(use_running_average=True), rngs=rngs)
        self.rngs = rngs

      def __call__(self, x):
        x = self.nn_dense1(x) + self.b
        return self.batchnorm(x)

    x = jax.random.normal(jax.random.key(0), (2, 4))
    model = NNXOuter(3, rngs=nnx.Rngs(0))
    gdef_before_lazy_init, _ = nnx.split(model)
    bridge.lazy_init(model, x)
    gdef_full, state = nnx.split(model)
    assert gdef_before_lazy_init != gdef_full
    assert 'nn_dense1' in state
    assert 'batchnorm' in state
    assert 'kernel' in state['nn_dense1']
    y = model(x)
    k, b = state['nn_dense1']['kernel'].value, state['b'].value
    np.testing.assert_allclose(y, x @ k + b, rtol=1e-5)
    assert gdef_full == nnx.graphdef(model)  # static data is stable now

  def test_linen_to_nnx_noncall_method(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        b = self.param('b', nn.zeros_init(), (1, 3,))
        return self.dot(x) + b

      @nn.compact
      def dot(self, x):
        w = self.param('w', nn.initializers.lecun_normal(), (4, 3))
        return x @ w

    x = jax.random.normal(jax.random.key(0), (2, 4))
    model = bridge.ToNNX(Foo(), rngs=nnx.Rngs(0))
    bridge.lazy_init(model, x, method=model.module.dot)
    y = model(x, method=model.module.dot)
    np.testing.assert_allclose(y, x @ nnx.state(model)['w'].value)
    # lazy_init only initialized param w inside dot(), so calling __call__ should fail
    with self.assertRaises(flax.errors.ScopeParamNotFoundError):
      y = model(x)

  def test_linen_to_nnx_mutable(self):
    class Foo(nn.Module):
      def setup(self):
        self.count = self.variable('counter', 'count', lambda: jnp.zeros((), jnp.int32))

      def __call__(self, x):
        if not self.is_initializing():
          self.count.value += 1
        return x

    x = lambda: jnp.zeros((), jnp.int32)
    model = bridge.ToNNX(Foo(), rngs=nnx.Rngs(0)).lazy_init(x)
    self.assertEqual(nnx.state(model)['count'].value, 0)
    y = model(x, mutable=True)
    self.assertEqual(nnx.state(model)['count'].value, 1)

  def test_linen_to_nnx_transform(self):
    class NNXOuter(nnx.Module):
      def __init__(self, dout: int, rngs: nnx.Rngs):
        self.inner = nnx.bridge.ToNNX(nn.Dense(dout), rngs=rngs)
        self.rngs = rngs

      def __call__(self, x):

        @nnx.split_rngs(splits=5)
        @nnx.vmap(in_axes=(0, None), axis_size=5)
        def vmap_fn(inner, x):
          return inner(x)

        return vmap_fn(self.inner, x)

    x = jax.random.normal(jax.random.key(0), (2, 4))
    model = NNXOuter(3, rngs=nnx.Rngs(0))
    nnx.bridge.lazy_init(model, x)

    self.assertEqual(model.inner.kernel.shape, (5, 4, 3))
    self.assertEqual(model.inner.bias.shape, (5, 3))

  def test_linen_to_nnx_metadata(self):
    linen_module = nn.Dense(
      features=64,
      kernel_init=nn.with_partitioning(nn.initializers.lecun_normal(), ('in', 'out')),
      bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ('out-alias',),
                                             rules=(('out-alias', 'out'),)),
      )
    x = jax.numpy.ones((1, 32))
    linen_vars = linen_module.init(jax.random.key(0), x)

    @nnx.jit
    def create_sharded_nnx_module(x):
      model = bridge.lazy_init(bridge.ToNNX(linen_module, rngs=nnx.Rngs(0)), x)
      state = nnx.state(model)
      sharded_state = nnx.with_sharding_constraint(state, nnx.get_partition_spec(state))
      nnx.update(model, sharded_state)
      return model
    with self.mesh:
      nnx_model = create_sharded_nnx_module(x)

    # nn.Partitioned metadata boxes translated into valid nnx.Variable boxes.
    self.assertIsInstance(linen_vars['params']['kernel'], nn.Partitioned)
    self.assertIsInstance(linen_vars['params']['bias'], nn.LogicallyPartitioned)
    self.assertIsInstance(nnx_model.kernel, nnx.Variable)
    assert nnx_model.kernel.sharding == ('in', 'out')
    assert nnx_model.kernel.value.sharding.is_equivalent_to(
      jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec('in', 'out')), ndim=2)

    assert nnx_model.bias.sharding == ('out-alias',)
    assert nnx_model.bias.sharding_rules == (('out-alias', 'out'),)
    assert nnx_model.bias.value.sharding.is_equivalent_to(
      jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec('out',)), ndim=1)

  def test_linen_to_nnx_state_structure_consistency(self):
    class LinenInner(nn.Module):
      dout: int
      @nn.compact
      def __call__(self, x):
        w = self.param('w', nn.initializers.lecun_normal(), (x.shape[-1], self.dout))
        return nn.Dropout(rate=0.5, deterministic=False)(x @ w)

    class LinenMiddle(nn.Module):
      dout: int
      @nn.compact
      def __call__(self, x):
        dot = LinenInner(self.dout, name='dot')
        b = self.variable('bias', 'b', nn.initializers.zeros_init(), None, (1, self.dout))
        return dot(x) + b.value

    @nnx.register_variable_name('bias')
    class Bias(nnx.Variable):
      pass

    class NNXMiddle(nnx.Module):
      def __init__(self, dout: int, *, rngs: nnx.Rngs):
        self.dot = bridge.ToNNX(LinenInner(dout), rngs=rngs)
        self.b = Bias(nnx.initializers.zeros_init()(rngs.params(), (1, dout)))
      def __call__(self, x):
        return self.dot(x) + self.b

    x = jax.random.normal(jax.random.key(42), (2, 4))
    from_top = bridge.lazy_init(
      bridge.ToNNX(LinenMiddle(dout=3), rngs=nnx.Rngs(0, dropout=1)), x)
    from_middle = bridge.lazy_init(
      NNXMiddle(dout=3, rngs=nnx.Rngs(0, dropout=1)), x)

    # Remove the NNX-module-local RNG states, which will be different
    # because the NNX modules are on different level
    def get_weights(model):
      return nnx.split(model, nnx.RngCount, nnx.RngKey, ...)[3]
    from_top_weights = get_weights(from_top)
    from_middle_weights = get_weights(from_middle)

    # Confirm the rest of the state has the same structure.
    self.assertEqual(jax.tree.structure(from_top_weights),
                     jax.tree.structure(from_middle_weights))

  def test_adding_new_attributes(self):
    class LinenModule(nn.Module):
      @nn.compact
      def __call__(self):
        if not self.is_initializing() and self.is_mutable_collection('cache'):
          self.put_variable('cache', 'x', 0)
        res = self.get_variable('cache', 'x')
        return res

    class NNXModule(nnx.Module):
      def __init__(self):
        self.module = nnx.bridge.ToNNX(LinenModule()).lazy_init()

      def __call__(self):
        result1 = self.module(mutable=['cache'])
        assert result1 == 0
        result2 = self.module()
        assert result2 == 0, result2  # fails: result2 is None

    module = NNXModule()
    module()

  ##################
  ### NNXToLinen ###
  ##################

  def test_nnx_to_linen(self):
    model = bridge.to_linen(nnx.Linear, 32, out_features=64)
    x = jax.numpy.ones((1, 32))
    y, variables = model.init_with_output(jax.random.key(0), x)
    assert y.shape == (1, 64)
    np.testing.assert_allclose(y, x @ variables['params']['kernel'])
    assert 'nnx' in variables
    assert isinstance(
      variables['nnx']['graphdef'], nnx.graph.NodeDef | nnx.graph.NodeRef
    )

  def test_nnx_to_linen_multiple_rngs(self):
    class NNXInner(nnx.Module):
      def __init__(self, din, dout, rngs):
        self.w = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (din, dout)))
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
      def __call__(self, x):
        return self.dropout(x @ self.w.value)

    class LinenOuter(nn.Module):
      @nn.compact
      def __call__(self, x):
        inner = bridge.to_linen(NNXInner, 4, 3)
        return inner(x)

    xkey, pkey, dkey1, dkey2 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(xkey, (2, 4))
    model = LinenOuter()
    y1, var = model.init_with_output({'params': pkey, 'dropout': dkey1}, x)
    y2 = model.apply(var, x, rngs={'dropout': dkey2})
    assert not jnp.allclose(y1, y2)  # dropout keys are different

  def test_nnx_to_linen_multiple_collections(self):
    class NNXInner(nnx.Module):
      def __init__(self, din, dout, rngs):
        self.w = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (din, dout)))
        self.bn = nnx.BatchNorm(dout, use_running_average=False, rngs=rngs)
        self.lora = nnx.LoRA(din, 3, dout, rngs=rngs)

      def __call__(self, x):
        return self.bn(x @ self.w.value) + self.lora(x)

    xkey, pkey, dkey = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(xkey, (2, 4))
    model = bridge.to_linen(NNXInner, 4, 3)
    var = model.init({'params': pkey, 'dropout': dkey}, x)
    self.assertSameElements(var.keys(), ['nnx', 'LoRAParam', 'params', 'batch_stats'])
    y = model.apply(var, x)
    assert y.shape == (2, 3)

  def test_nnx_to_linen_mutable(self):

    @nnx.register_variable_name('Count', overwrite=True)
    class Count(nnx.Variable):
      pass

    class Counter(nnx.Module):
      def __init__(self):
        self.count = Count(jnp.array(0))
      def __call__(self):
        self.count += 1

    model = bridge.ToLinen(Counter, skip_rng=True)
    variables = model.init(jax.random.key(0))
    assert variables['Count']['count'] == 0

    _, updates = model.apply(variables, mutable='Count')
    assert updates['Count']['count'] == 1
    _ = model.apply(variables | updates)

  def test_nnx_to_linen_mutated_static_data(self):

    @nnx.register_variable_name('Count', overwrite=True)
    class Count(nnx.Variable):
      pass

    class Counter(nnx.Module):
      def __init__(self):
        self.count = Count(jnp.array(0))
      def __call__(self):
        self.count += 1
        self.count_nonzero = Count(jnp.array(1))

    model = bridge.ToLinen(Counter, skip_rng=True)
    variables = model.init(jax.random.key(0))
    assert variables['Count']['count'] == 0

    # This does not work, because the __call__ also changes the static data of the model.
    _, updates = model.apply(variables, mutable='Count')
    assert updates['Count']['count'] == 1
    assert updates['Count']['count_nonzero'] == 1
    with self.assertRaises(ValueError):
      _ = model.apply(variables | updates)

    # This makes sure the static data is updated too. Using mutable=True also works.
    _, updates = model.apply(variables, mutable=['Count', 'nnx'])
    assert updates['Count']['count'] == 1
    assert updates['Count']['count_nonzero'] == 1
    _ = model.apply(variables | updates)

  def test_nnx_to_linen_transforms(self):
    class LinenOuter(nn.Module):
      dout: int
      @nn.compact
      def __call__(self, x):
        inner = nn.vmap(
          bridge.ToLinen,
          variable_axes={'params': 0, 'nnx': None},
          split_rngs={'params': True}
        )(nnx.Linear, args=(x.shape[-1], self.dout))
        return inner(x)

    xkey, pkey, _ = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(xkey, (2, 4))
    model = LinenOuter(dout=3)
    y, var = model.init_with_output(pkey, x)
    k = var['params']['VmapToLinen_0']['kernel']
    assert k.shape == (2, 4, 3)
    np.testing.assert_allclose(y, jnp.einsum('ab,abc->ac', x, k))
    assert 'nnx' in var

  def test_nnx_to_linen_metadata(self):
    model = bridge.to_linen(
      nnx.Linear, 32, 64,
      kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ('in', 'out')))
    x = jax.numpy.ones((1, 32))
    y, variables = model.init_with_output(jax.random.key(0), x)
    assert y.shape == (1, 64)
    self.assertIsInstance(variables['params']['kernel'], nnx.bridge.NNXMeta)
    assert variables['params']['kernel'].metadata['sharding'] == ('in', 'out')
    self.assertEqual(nn.get_partition_spec(variables)['params']['kernel'],
                     jax.sharding.PartitionSpec('in', 'out'))
    np.testing.assert_allclose(y, x @ variables['params']['kernel'].value)

  def test_nnx_to_linen_metadata_transform(self):
    # TODO: add support and testing after axis add/remove in transform is fixed.
    pass

  def test_nnx_to_linen_pytree_structure_consistency(self):
    class NNXInner(nnx.Module):
      def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        self.w = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (din, dout)))
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
      def __call__(self, x):
        return self.dropout(x @ self.w)

    @nnx.register_variable_name('bias', overwrite=True)
    class Bias(nnx.Variable):
      pass

    class NNXMiddle(nnx.Module):
      def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        self.dot = NNXInner(din, dout, rngs=rngs)
        self.b = Bias(nnx.initializers.zeros_init()(rngs.params(), (1, dout)))
      def __call__(self, x):
        return self.dot(x) + self.b

    class LinenMiddle(nn.Module):
      dout: int
      @nn.compact
      def __call__(self, x):
        dot = bridge.to_linen(NNXInner, x.shape[-1], self.dout, name='dot')
        b = self.variable('bias', 'b', nn.initializers.zeros_init(), None, (1, self.dout))
        return dot(x) + b.value

    x = jax.random.normal(jax.random.key(42), (2, 4))
    keys = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
    from_top = bridge.to_linen(NNXMiddle, din=4, dout=3).init(keys, x)
    from_middle = LinenMiddle(dout=3).init(keys, x)

    # Remove the NNX-module-local RNG states, which will be different
    # because the NNX modules are on different level
    def get_weights(variables):
      non_rngs = {}
      for kp, v in flax.traverse_util.flatten_dict(variables).items():
        if 'rngs' not in kp and 'nnx' not in kp:
          non_rngs[kp] = v
      return flax.traverse_util.unflatten_dict(non_rngs)
    from_top_weights = get_weights(from_top)
    from_middle_weights = get_weights(from_middle)

    # Confirm the rest of the state has the same structure.
    self.assertEqual(jax.tree.structure(from_top_weights),
                     jax.tree.structure(from_middle_weights))

  ############################
  ### Hybrid mix-and-match ###
  ############################

  def test_nnx_linen_nnx(self):
    class NNXInner(nnx.Module):
      def __init__(self, din, dout, dropout_rate, rngs):
        self.w = nnx.Param(
          nnx.with_partitioning(nnx.initializers.lecun_normal(), sharding=('in', 'out')
                                )(rngs.params(), (din, dout)))
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
      def __call__(self, x):
        return self.dropout(x @ self.w.value)

    class LinenMiddle(nn.Module):
      dout: int
      dropout_rate: float
      @nn.compact
      def __call__(self, x):
        dot = bridge.to_linen(NNXInner, x.shape[-1], self.dout, self.dropout_rate, name='dot')
        logical_init = nn.with_logical_partitioning(
          nn.initializers.lecun_normal(), ('out-alias',), rules=(('out-alias', 'out')))
        b = self.param('b', logical_init, (1, self.dout))
        return dot(x) + b

    class NNXOuter(nnx.Module):
      def __init__(self, dout: int, dropout_rate: float, *, rngs: nnx.Rngs):
        self.inner = bridge.ToNNX(LinenMiddle(dout, dropout_rate), rngs=rngs)
        self.rngs = rngs
      def __call__(self, x):
        return self.inner(x)

    x = jax.random.normal(jax.random.key(0), (2, 4))

    # Test the RNG
    model = bridge.lazy_init(NNXOuter(dout=3, dropout_rate=0.5,
                                      rngs=nnx.Rngs(default=1, dropout=2)), x)
    y1, y2 = model(x), model(x)
    # The dropout key of lowest NNX level still changes over stateful calls
    assert not jnp.allclose(y1, y2)
    # Reseed resets the RNG key back
    nnx.reseed(model, dropout=2)
    np.testing.assert_array_equal(y1, model(x))

    # Test the param value with disabled dropout
    model = bridge.lazy_init(NNXOuter(dout=3, dropout_rate=0.,
                                      rngs=nnx.Rngs(default=1, dropout=2)), x)
    w, b = model.inner.dot['w'], model.inner.b
    self.assertIsInstance(w, nnx.Param)
    np.testing.assert_allclose(model(x), x @ w + b)
    assert hasattr(w, 'sharding') and w.sharding == ('in', 'out')
    assert hasattr(b, 'sharding') and b.sharding == ('out-alias', )

  def test_linen_nnx_linen(self):
    # TODO: add when we can safely `lazy_init` the NNX module inside `ToLinen` without
    # messing up the stateful part of the NNX module.
    pass

class TestCompatModule(absltest.TestCase):
  def test_update(self):
    class Foo(bridge.Module):
      a: int

    foo = Foo(1)
    state = {'b': {'c': nnx.Param(jnp.array(2))}}
    nnx.update(foo, state)

  def test_module_stack(self):
    """Test that apply set the module stack correctly."""
    test = self

    class Foo(bridge.Module):
      def setup(self):
        current_ctx = MODULE_CONTEXT.module_stack[-1]
        test.assertIs(current_ctx.module, self)
        test.assertFalse(current_ctx.in_compact)

      def __call__(self):
        current_ctx = MODULE_CONTEXT.module_stack[-1]
        test.assertIs(current_ctx.module, self)
        test.assertFalse(current_ctx.in_compact)

    foo = Foo()
    foo.apply({})

  def test_compact_basic(self):
    test = self
    class Linear(bridge.Module):
      dout: int

      @bridge.compact
      def __call__(self, x):
        w = self.param(
          'w', nnx.initializers.uniform(), (x.shape[-1], self.dout)
        )
        b = self.param('b', nn.initializers.zeros_init(), (self.dout,))
        return x @ w + b[None]

    class Foo(bridge.Module):
      dout: int

      @bridge.compact
      def __call__(self, x):
        din = x.shape[-1]
        self.linear = Linear(self.dout)
        x = self.linear(x)

        # NNX
        graphdef, state = nnx.split(self)
        test.assertIn('Linear_0', state)
        test.assertIn('w', state['Linear_0'])
        test.assertIn('b', state['Linear_0'])

        return x

    foo = Foo(5)
    x = jnp.ones((3, 2))

    self.assertIsInstance(foo, nnx.Module)

    variables = foo.init(0, x)
    params = variables['params']

    self.assertIn('Linear_0', params)
    self.assertIn('w', params['Linear_0'])
    self.assertIn('b', params['Linear_0'])
    self.assertEqual(params['Linear_0']['w'].shape, (2, 5))
    self.assertEqual(params['Linear_0']['b'].shape, (5,))

    y: jax.Array = foo.apply(variables, x)

    self.assertEqual(y.shape, (3, 5))

  def test_mutable_state(self):
    class FooLinen(nn.Module):
      @nn.compact
      def __call__(self):
        count = self.variable(
          'counts', 'count', lambda: jnp.zeros((), jnp.int32)
        )
        count.value += 1

    model_linen = FooLinen()
    initial_vars_linen = model_linen.init({})
    _, vars_linen = model_linen.apply(initial_vars_linen, mutable='counts')

    class FooNNX(bridge.Module):
      @bridge.compact
      def __call__(self):
        count = self.variable(
          'counts', 'count', lambda: jnp.zeros((), jnp.int32)
        )
        count.value += 1

    model_nnx = FooNNX()

    initial_vars_nnx = model_nnx.init({})
    _, vars_nnx = model_nnx.apply(initial_vars_nnx, mutable='counts')

    self.assertEqual(
      initial_vars_linen['counts']['count'], initial_vars_nnx['counts']['count']
    )
    self.assertEqual(vars_linen['counts']['count'], vars_nnx['counts']['count'])

  def test_compact_parent_none(self):
    class Foo(bridge.Module):
      pass

    class Bar(bridge.Module):
      @bridge.compact
      def __call__(self):
        return Foo().scope

    bar = Bar()
    scope = bar.apply({}, rngs=1)
    self.assertIsNone(bar.scope)

    self.assertEqual(scope.rngs.default.key.value, jax.random.key(1))
    self.assertEqual(scope.rngs.default.count.value, 0)

    class Baz(bridge.Module):
      @bridge.compact
      def __call__(self):
        return Foo(parent=None).scope

    baz = Baz()
    scope = baz.apply({}, rngs=1)
    self.assertIsNone(scope)

  def test_name(self):
    class Foo(bridge.Module):
      dout: int

      def __call__(self, x):
        w = self.param(
          'w', nnx.initializers.uniform(), (x.shape[-1], self.dout)
        )
        return x @ w

    class Bar(bridge.Module):
      @bridge.compact
      def __call__(self, x):
        return Foo(5, name='xyz')(x)

    bar = Bar()
    x = jnp.ones((1, 2))
    y, variables = bar.init_with_output(0, x)

    self.assertIn('xyz', variables['params'])
    self.assertEqual(variables['params']['xyz']['w'].shape, (2, 5))
    self.assertEqual(y.shape, (1, 5))

    y = bar.apply(variables, x)
    self.assertEqual(y.shape, (1, 5))

    with self.assertRaises(ValueError):
      class SetupBar(bridge.Module):
        def setup(self):
          self.xyz = Foo(5, name='xyz')
        def __call__(self, x):
          return self.xyz(x)
      SetupBar().init(0, x)

  def test_dense_port(self):
    class Dense(bridge.Module):
      features: int
      use_bias: bool = True
      dtype: Any = None
      param_dtype: Any = jnp.float32
      precision: Any = None
      kernel_init: Any = nnx.initializers.lecun_normal()
      bias_init: Any = nnx.initializers.zeros_init()
      # Deprecated. Will be removed.
      dot_general: Any | None = None
      dot_general_cls: Any = None

      @bridge.compact
      def __call__(self, inputs: jax.Array) -> jax.Array:
        kernel = self.param(
          'kernel',
          self.kernel_init,
          (jnp.shape(inputs)[-1], self.features),
          self.param_dtype,
        )
        if self.use_bias:
          bias = self.param(
            'bias', self.bias_init, (self.features,), self.param_dtype
          )
        else:
          bias = None
        inputs, kernel, bias = promote_dtype(
          inputs, kernel, bias, dtype=self.dtype
        )

        if self.dot_general_cls is not None:
          dot_general = self.dot_general_cls()
        elif self.dot_general is not None:
          dot_general = self.dot_general
        else:
          dot_general = jax.lax.dot_general
        y = dot_general(
          inputs,
          kernel,
          (((inputs.ndim - 1,), (0,)), ((), ())),
          precision=self.precision,
        )
        if bias is not None:
          y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y

    m = Dense(3)
    x = jnp.ones((1, 10, 2))
    y, variables = m.init_with_output(0, x)

    self.assertEqual(y.shape, (1, 10, 3))
    self.assertEqual(variables['params']['kernel'].shape, (2, 3))
    self.assertEqual(variables['params']['bias'].shape, (3,))

    y = m.apply(variables, x)

    self.assertEqual(y.shape, (1, 10, 3))
    self.assertEqual(variables['params']['kernel'].shape, (2, 3))
    self.assertEqual(variables['params']['bias'].shape, (3,))

    @jax.jit
    def train_step(params, x, y):
      def loss_fn(params):
        y_pred = m.apply({'params': params}, x)
        return jnp.mean((y - y_pred) ** 2)

      grads = jax.grad(loss_fn)(params)

      params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)

      return params

    params = variables['params']
    x = jnp.ones((1, 10, 2))
    y = jnp.ones((1, 10, 3))

    params = train_step(params, x, y)

  def test_metadata(self):
    class Linear(bridge.Module):
      dout: int

      @bridge.compact
      def __call__(self, x):
        w = self.param(
          'w', bridge.with_partitioning(nnx.initializers.uniform(), ('in', 'out')),
          (x.shape[-1], self.dout)
        )
        b = self.param('b', nnx.initializers.zeros_init(), (self.dout,))
        return x @ w + b[None]

    foo = Linear(5)
    x = jnp.ones((3, 2))

    variables = foo.init(0, x)
    params = variables['params']
    self.assertIsInstance(params['w'], nn.Partitioned)
    self.assertEqual(params['w'].value.shape, (2, 5))
    self.assertEqual(params['w'].names, ('in', 'out'))
    self.assertEqual(nn.get_partition_spec(variables)['params']['w'],
                     jax.sharding.PartitionSpec('in', 'out'))
    self.assertIsInstance(params['b'], jax.Array)
    self.assertEqual(params['b'].shape, (5,))

    y: jax.Array = foo.apply(variables, x)
    self.assertEqual(y.shape, (3, 5))


if __name__ == '__main__':
  absltest.main()
