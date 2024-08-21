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

from functools import partial

from absl.testing import absltest
import flax
from flax import linen as nn
from flax import nnx
from flax.nnx import bridge
import jax
import jax.numpy as jnp
import numpy as np


class TestCompatibility(absltest.TestCase):
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
    assert 'params' in state.nn_dense1
    assert 'batch_stats' in state.batchnorm
    y = model(x)
    k, b = state.nn_dense1.params.kernel.value, state.b.value
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
    np.testing.assert_allclose(y, x @ nnx.state(model).params.w.value)
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
    assert nnx.state(model).counter.count.value == 0
    y = model(x, mutable=True)
    assert nnx.state(model).counter.count.value == 1

  def test_linen_to_nnx_transform(self):
    class NNXOuter(nnx.Module):
      def __init__(self, dout: int, rngs: nnx.Rngs):
        self.inner = nnx.bridge.ToNNX(nn.Dense(dout), rngs=rngs)
        self.rngs = rngs

      def __call__(self, x):
        @partial(nnx.vmap, in_axes=None, state_axes={...: 0}, axis_size=5)
        def vmap_fn(inner, x):
          return inner(x)

        return vmap_fn(self.inner, x)

    x = jax.random.normal(jax.random.key(0), (2, 4))
    model = NNXOuter(3, rngs=nnx.Rngs(0))
    nnx.bridge.lazy_init(model, x)

    self.assertEqual(model.inner.params['kernel'].shape, (5, 4, 3))
    self.assertEqual(model.inner.params['bias'].shape, (5, 3))

  ##################
  ### NNXToLinen ###
  ##################

  def test_nnx_to_linen(self):
    model = bridge.to_linen(nnx.Linear, 32, out_features=64)
    x = jax.numpy.ones((1, 32))
    y, variables = model.init_with_output(jax.random.key(0), x)
    assert y.shape == (1, 64)
    np.testing.assert_allclose(y, x @ variables['params']['kernel'].value)
    assert 'nnx' in variables
    assert isinstance(variables['nnx']['graphdef'], nnx.GraphDef)

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
    nnx.register_variable_name_type_pair('Count', Count)

    class Counter(nnx.Module):
      def __init__(self):
        self.count = Count(jnp.array(0))
      def __call__(self):
        self.count += 1

    model = bridge.ToLinen(Counter, skip_rng=True)
    variables = model.init(jax.random.key(0))
    assert variables['Count']['count'].value == 0

    _, updates = model.apply(variables, mutable='Count')
    assert updates['Count']['count'].value == 1
    _ = model.apply(variables | updates)

  def test_nnx_to_linen_mutated_static_data(self):
    class Count(nnx.Variable): pass
    nnx.register_variable_name_type_pair('Count', Count)

    class Counter(nnx.Module):
      def __init__(self):
        self.count = Count(jnp.array(0))
      def __call__(self):
        self.count += 1
        self.count_nonzero = Count(jnp.array(1))

    model = bridge.ToLinen(Counter, skip_rng=True)
    variables = model.init(jax.random.key(0))
    assert variables['Count']['count'].value == 0

    # This does not work, because the __call__ also changes the static data of the model.
    _, updates = model.apply(variables, mutable='Count')
    assert updates['Count']['count'].value == 1
    assert updates['Count']['count_nonzero'].value == 1
    with self.assertRaises(ValueError):
      _ = model.apply(variables | updates)

    # This makes sure the static data is updated too. Using mutable=True also works.
    _, updates = model.apply(variables, mutable=['Count', 'nnx'])
    assert updates['Count']['count'].value == 1
    assert updates['Count']['count_nonzero'].value == 1
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
    k = var['params']['VmapToLinen_0']['kernel'].value
    assert k.shape == (2, 4, 3)
    np.testing.assert_allclose(y, jnp.einsum('ab,abc->ac', x, k))
    assert 'nnx' in var

  ############################
  ### Hybrid mix-and-match ###
  ############################

  def test_nnx_linen_nnx(self):
    class NNXInner(nnx.Module):
      def __init__(self, din, dout, rngs):
        self.w = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (din, dout)))
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
      def __call__(self, x):
        return self.dropout(x @ self.w.value)

    class LinenMiddle(nn.Module):
      dout: int
      @nn.compact
      def __call__(self, x):
        dot = bridge.to_linen(NNXInner, x.shape[-1], self.dout, name='linen')
        b = self.param('b', nn.zeros_init(), (1, self.dout))
        return dot(x) + b

    class NNXOuter(nnx.Module):
      def __init__(self, dout: int, *, rngs: nnx.Rngs):
        self.inner = bridge.ToNNX(LinenMiddle(dout), rngs=rngs)
        self.rngs = rngs
      def __call__(self, x):
        return self.inner(x)

    x = jax.random.normal(jax.random.key(0), (2, 4))
    model = bridge.lazy_init(NNXOuter(3, rngs=nnx.Rngs(default=1, dropout=2)), x)
    y1, y2 = model(x), model(x)
    # The dropout key of lowest NNX level still changes over stateful calls
    assert not jnp.allclose(y1, y2)
    # Reseed resets the RNG key back
    nnx.reseed(model, dropout=2)
    np.testing.assert_array_equal(y1, model(x))

if __name__ == '__main__':
  absltest.main()
