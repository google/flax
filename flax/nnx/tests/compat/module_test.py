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

import dataclasses

from absl.testing import absltest
import jax
import jax.numpy as jnp

from flax import nnx
from flax.nnx import compat


class TestCompatModule(absltest.TestCase):
  def test_compact_basic(self):
    class Linear(compat.Module):
      dout: int

      def setup(self):
        self.count = 0

      def __call__(self, x):
        self.count += 1
        if not hasattr(self, 'w'):
          assert self.scope is not None
          rngs = self.scope.rngs
          self.w = nnx.Param(
            jax.random.uniform(rngs(), (x.shape[-1], self.dout))
          )
          self.b = nnx.Param(jnp.zeros((self.dout,)))
        return x @ self.w + self.b[None]

    @dataclasses.dataclass
    class Foo(compat.Module):
      dout: int

      @compat.compact
      def __call__(self, x):
        din = x.shape[-1]
        self.linear = Linear(self.dout)
        x = self.linear(x)
        return x

    foo = Foo(5)
    x = jnp.ones((3, 2))
    rngs = nnx.Rngs(0)

    foo._set_scope(compat.Scope(rngs))
    y = foo(x)
    foo._set_scope(None)

    assert y.shape == (3, 5)
    assert hasattr(foo, 'Linear_0')

    assert foo.linear is foo.Linear_0
    assert foo.linear.count == 1
    assert rngs.default.count.value == 1

    foo._set_scope(compat.Scope(rngs))
    y = foo(x)
    foo._set_scope(None)

    assert foo.linear is foo.Linear_0
    assert foo.linear.count == 2

    # Rngs not called again
    assert rngs.default.count.value == 1

  def test_compact_parent_none(self):
    class Foo(compat.Module):
      pass

    class Bar(compat.Module):
      @compat.compact
      def __call__(self):
        return Foo().scope

    rngs = nnx.Rngs(0)
    bar = Bar()
    bar._set_scope(compat.Scope(rngs))
    scope = bar()
    bar._set_scope(None)
    assert bar.scope is None
    assert scope.rngs is rngs

    class Baz(compat.Module):
      @compat.compact
      def __call__(self):
        return Foo(parent=None).scope

    baz = Baz()
    baz._set_scope(compat.Scope(rngs))
    scope = baz()
    baz._set_scope(None)
    assert scope is None

  def test_name(self):
    class Foo(compat.Module):
      dout: int

      def __call__(self, x):
        if not hasattr(self, 'w'):
          assert self.scope is not None
          rngs = self.scope.rngs
          self.w = nnx.Param(
            jax.random.uniform(rngs(), (x.shape[-1], self.dout))
          )
        return x @ self.w

    class Bar(compat.Module):
      @compat.compact
      def __call__(self, x):
        return Foo(5, name='foo')(x)

    bar = Bar()
    x = jnp.ones((1, 2))
    rngs = nnx.Rngs(0)
    bar._set_scope(compat.Scope(rngs))
    y = bar(x)
    bar._set_scope(None)
    assert y.shape == (1, 5)

    assert hasattr(bar, 'foo')
    assert isinstance(bar.foo, Foo)

if __name__ == '__main__':
  absltest.main()
