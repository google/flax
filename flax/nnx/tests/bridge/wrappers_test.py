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

  def test_linen_to_nnx(self):
    ## Wrapper API for Linen Modules
    linen_module = nn.Dense(features=64)
    x = jax.numpy.ones((1, 32))
    model = bridge.LinenToNNX(linen_module, rngs=nnx.Rngs(0)).lazy_init(x)  # like linen init
    y = model(x)  # like linen apply
    assert y.shape == (1, 64)

  def test_linen_to_nnx_submodule(self):
    class NNXOuter(nnx.Module):
      def __init__(self, dout: int, *, rngs: nnx.Rngs):
        self.nn_dense1 = bridge.LinenToNNX(nn.Dense(dout, use_bias=False), rngs=rngs)
        self.b = nnx.Param(jax.random.uniform(rngs.params(), (1, dout,)))
        self.batchnorm = bridge.LinenToNNX(nn.BatchNorm(use_running_average=True), rngs=rngs)
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
    model = bridge.LinenToNNX(Foo(), rngs=nnx.Rngs(0))
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
    model = bridge.LinenToNNX(Foo(), rngs=nnx.Rngs(0)).lazy_init(x)
    assert nnx.state(model).counter.count.value == 0
    y = model(x, mutable=True)
    assert nnx.state(model).counter.count.value == 1


if __name__ == '__main__':
  absltest.main()
