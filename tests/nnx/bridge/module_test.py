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
from absl.testing import absltest
import numpy as np

from flax import linen as nn
from flax import nnx
from flax.nnx import bridge
from flax.nnx.bridge.module import MODULE_CONTEXT


class TestBridgeModule(absltest.TestCase):
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

  def test_pure_nnx_submodule(self):
    class NNXLayer(nnx.Module):
      def __init__(self, dim, dropout, rngs):
        self.linear = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.count = nnx.Intermediate(jnp.array([0.]))
      def __call__(self, x):
        # Required check to avoid state update in `init()`. Can this be avoided?
        if not bridge.current_module().is_initializing():
          self.count.value = self.count.value + 1
        x = self.linear(x)
        x = self.dropout(x)
        return x

    class BridgeMLP(bridge.Module):
      @bridge.compact
      def __call__(self, x):
        x = bridge.nnx_in_bridge_mdl(lambda r: NNXLayer(8, 0.3, rngs=r))(x)
        x = bridge.nnx_in_bridge_mdl(
          lambda r: NNXLayer(8, 0.3, rngs=r), name='another')(x)
        return x

    model = BridgeMLP()
    x = jax.random.normal(jax.random.key(0), (4, 8))
    variables = model.init(jax.random.key(1), x)
    self.assertSameElements(variables['params'].keys(),
                            ['NNXLayer_0', 'another'])
    self.assertFalse(jnp.array_equal(
      variables['params']['NNXLayer_0']['linear']['kernel'],
      variables['params']['another']['linear']['kernel'], ))
    self.assertEqual(variables['intermediates']['NNXLayer_0']['count'], 0)

    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    y1 = model.apply(variables, x, rngs={'params': k1, 'dropout': k2})
    y2 = model.apply(variables, x, rngs={'params': k1, 'dropout': k3})
    assert not jnp.array_equal(y1, y2)

    _, updates = model.apply(variables, x, rngs={'params': k1, 'dropout': k3},
                             mutable=True)
    self.assertEqual(updates['intermediates']['NNXLayer_0']['count'], 1)

    class BridgeMLPSetup(bridge.Module):
      def setup(self):
        self.layer = bridge.nnx_in_bridge_mdl(
          lambda r: NNXLayer(8, 0.3, rngs=r))
      def __call__(self, x):
        return self.layer(x)

    model = BridgeMLPSetup()
    variables = model.init(jax.random.key(1), x)
    self.assertSameElements(variables['params'].keys(), ['layer'])
    y1 = model.apply(variables, x, rngs={'params': k1, 'dropout': k2})
    y2 = model.apply(variables, x, rngs={'params': k1, 'dropout': k3})
    assert not jnp.array_equal(y1, y2)

  def test_pure_nnx_submodule_modified_rng(self):
    class FooStack(nnx.Module):
      def __init__(self, in_dim, key):
        keys = jax.random.split(key, in_dim)
        self.rngs = nnx.Rngs(keys)
      def __call__(self, x):
        @nnx.vmap
        def generate_weights(r):
          return jax.random.normal(r.default(), (2,))
        w = generate_weights(self.rngs)
        return x @ w

    class BridgeFoo(bridge.Module):
      @bridge.compact
      def __call__(self, x):
        x = bridge.nnx_in_bridge_mdl(lambda r: FooStack(4, r.default()))(x)
        return x

    model = BridgeFoo()
    v = model.init(jax.random.key(1), jnp.ones((1, 4)))
    y = model.apply(v, jnp.ones((1, 4)), rngs=jax.random.key(1))

  def test_linen_submodule(self):
    class LinenLayer(nn.Module):
      dim: int
      dropout_rate: float
      def setup(self):
        self.linear = nn.Dense(self.dim, use_bias=False)
        self.dropout = nn.Dropout(self.dropout_rate, deterministic=False)

      def __call__(self, x):
        if not self.is_initializing():
          self.sow('intermediates', 'count', 1,
                    init_fn=lambda: 0, reduce_fn=lambda a, b: a + b)
        x = self.linear(x)
        x = self.dropout(x)
        return x

    class BridgeMLP(bridge.Module):
      @bridge.compact
      def __call__(self, x):
        x = bridge.linen_in_bridge_mdl(LinenLayer(8, 0.3))(x)
        x = bridge.linen_in_bridge_mdl(LinenLayer(8, 0.3), name='another')(x)
        return x

    model = BridgeMLP()
    x = jax.random.normal(jax.random.key(0), (4, 8))
    variables = model.init(jax.random.key(1), x)
    self.assertFalse(jnp.array_equal(
      variables['params']['LinenLayer_0']['linear']['kernel'],
      variables['params']['another']['linear']['kernel'], ))

    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    y1 = model.apply(variables, x, rngs={'params': k1, 'dropout': k2})
    y2 = model.apply(variables, x, rngs={'params': k1, 'dropout': k3})
    assert not jnp.array_equal(y1, y2)

    _, updates = model.apply(variables, x, rngs={'params': k1, 'dropout': k3},
                             mutable=True)
    self.assertEqual(updates['intermediates']['LinenLayer_0']['count'], 1)

    class BridgeMLPSetup(bridge.Module):
      def setup(self):
        self.layer = bridge.linen_in_bridge_mdl(LinenLayer(8, 0.3))
      def __call__(self, x):
        return self.layer(x)

    model = BridgeMLPSetup()
    variables = model.init(jax.random.key(1), x)
    self.assertSameElements(variables['params'].keys(), ['layer'])
    y1 = model.apply(variables, x, rngs={'params': k1, 'dropout': k2})
    y2 = model.apply(variables, x, rngs={'params': k1, 'dropout': k3})
    assert not jnp.array_equal(y1, y2)

  def test_name(self):
    class Foo(bridge.Module):
      name: str

    class Bar(bridge.Module):
      @bridge.compact
      def __call__(self):
        f = Foo(name='f')
        assert f.name == 'f'
        assert self.f == f

    Bar().init()

  def test_transforms(self):
    class Dense(bridge.Module):
      dout: int
      @bridge.compact
      def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.param('w', nn.initializers.normal(),
                              (x.shape[-1], self.dout))

    class MLP(bridge.Module):
      dim: int
      num_layers: int
      def setup(self):
        @nnx.split_rngs(splits=self.num_layers)
        @nnx.vmap(
            in_axes=(nnx.StateAxes({nnx.RngState: 0, ...: None}),),
            axis_size=self.num_layers,
            transform_metadata={nnx.PARTITION_NAME: None},
        )
        def create_block(parent):
          block = Dense(self.dim)
          parent.block = block
        create_block(self)

      def __call__(self, x):
        @nnx.split_rngs(splits=self.num_layers)
        @nnx.scan(
            in_axes=(0, nnx.Carry),
            out_axes=nnx.Carry,
            transform_metadata={nnx.PARTITION_NAME: None},
        )
        def forward_block(model, x):
          return model(x)
        x = forward_block(self.block, x)
        return x

    model = MLP(dim=32, num_layers=2)
    x = jnp.ones((4, 32))
    variables = model.init(jax.random.key(0), x)
    y = model.apply(variables, x)
    w = variables['params']['block']['w']
    np.testing.assert_array_equal(y, x @ w[0] @ w[1])

  def test_shared_modules(self):
    class Dense(bridge.Module):
      dout: int
      @bridge.compact
      def __call__(self, x):
        return x @ self.param('w', nn.initializers.normal(),
                              (x.shape[-1], self.dout))

    class Bottom(bridge.Module):
      def setup(self):
        self.layer = Dense(4)

      def __call__(self, x):
        return self.layer(x)

    class Top(bridge.Module):
      def setup(self):
        self.zzz = Bottom()
        self.set_attr_priority('zzz', bridge.AttrPriority.HIGH)
        self.aaa = self.zzz  # another reference
        self.dense = self.aaa.layer  # and another reference

      def forward(self, x):
        return self.aaa(x)

      def __call__(self, x):
        forward = nnx.remat(self.__class__.forward)
        return forward(self, x)

    model = Top()
    x = jnp.ones((4, 32))
    params = model.init(jax.random.key(0), x)['params']
    self.assertSameElements(['zzz'], params.keys())

  def test_linen_layer_naming(self):
    class Dense(bridge.Module):
      dout: int
      @bridge.compact
      def __call__(self, x):
        return x @ self.param('w', lambda _: jnp.ones((x.shape[-1], self.dout)))

    class MLP(bridge.Module):
      nlayers: int
      def setup(self):
        self.layers = [Dense(4, name=f'layer_{i}') for i in range(self.nlayers)]
      def __call__(self, x):
        for layer in self.layers:
          x = layer(x)
        return x

    model = MLP(nlayers=3)
    x = jnp.ones((2, 4))
    params = model.init(jax.random.key(0), x)['params']
    self.assertSameElements([f'layer_{i}' for i in range(3)], params.keys())


if __name__ == '__main__':
  absltest.main()

