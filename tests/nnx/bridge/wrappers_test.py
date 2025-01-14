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
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

from absl.testing import absltest
import flax
from flax import linen as nn
from flax import nnx
from flax.nnx import bridge
import jax
import jax.numpy as jnp
import numpy as np


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
    assert 'kernel' in state.nn_dense1
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
    np.testing.assert_allclose(y, x @ nnx.state(model).w.value)
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
    self.assertEqual(nnx.state(model).count.value, 0)
    y = model(x, mutable=True)
    self.assertEqual(nnx.state(model).count.value, 1)

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

    class Bias(nnx.Variable): pass
    nnx.register_variable_name_type_pair('bias', Bias)
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
    class Count(nnx.Variable): pass
    nnx.register_variable_name_type_pair('Count', Count, overwrite=True)

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
    class Count(nnx.Variable): pass
    nnx.register_variable_name_type_pair('Count', Count, overwrite=True)

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

    class Bias(nnx.Variable): pass
    nnx.register_variable_name_type_pair('bias', Bias, overwrite=True)
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


if __name__ == '__main__':
  absltest.main()
