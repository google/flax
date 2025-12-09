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

from flax import nnx
import jax
from jax import numpy as jnp


class StateTest(absltest.TestCase):
  def test_create_state(self):
    state = nnx.State(
      {'a': nnx.Param(jnp.array(1)), 'b': {'c': nnx.Param(jnp.array(2))}}
    )

    assert state['a'][...] == 1
    assert state['b']['c'][...] == 2

  def test_get_attr(self):
    state = nnx.State(
      {'a': nnx.Param(jnp.array(1)), 'b': {'c': nnx.Param(jnp.array(2))}}
    )

    assert state.a[...] == 1
    assert state.b.c[...] == 2

  def test_set_attr(self):
    state = nnx.State(
      {'a': nnx.Param(jnp.array(1)), 'b': {'c': nnx.Param(jnp.array(2))}}
    )

    state.a[...] = 3
    state.b.c[...] = 4

    assert state['a'][...] == 3
    assert state['b']['c'][...] == 4

  def test_set_attr_variables(self):
    state = nnx.State(
      {'a': nnx.Param(jnp.array(1)), 'b': {'c': nnx.Param(jnp.array(2))}}
    )

    state.a[...] = 3
    state.b.c[...] = 4

    assert isinstance(state.a, nnx.Param)
    assert state.a[...] == 3
    assert isinstance(state.b.c, nnx.Param)
    assert state.b.c[...] == 4

  def test_add_nested_attr(self):
    state = nnx.State(
      {'a': nnx.Param(jnp.array(1)), 'b': {'c': nnx.Param(jnp.array(2))}}
    )
    state.b.d = nnx.Param(jnp.array(5))

    assert state['b']['d'][...] == 5

  def test_delete_nested_attr(self):
    state = nnx.State(
      {'a': nnx.Param(jnp.array(1)), 'b': {'c': nnx.Param(jnp.array(2))}}
    )
    del state['b']['c']

    assert 'c' not in state['b']

  def test_integer_access(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.layers = nnx.List([
          nnx.Linear(1, 2, rngs=rngs),
          nnx.Linear(2, 3, rngs=rngs)
        ])

    module = Foo(rngs=nnx.Rngs(0))
    state = nnx.state(module)

    assert module.layers[0].kernel.shape == (1, 2)
    assert state.layers[0].kernel.shape == (1, 2)
    assert module.layers[1].kernel.shape == (2, 3)
    assert state.layers[1].kernel.shape == (2, 3)

  def test_pure_dict(self):
    module = nnx.Linear(4, 5, rngs=nnx.Rngs(0))
    state = nnx.state(module)
    pure_dict = nnx.to_pure_dict(state)
    assert isinstance(pure_dict, dict)
    assert isinstance(pure_dict['kernel'], jax.Array)
    assert isinstance(pure_dict['bias'], jax.Array)
    nnx.replace_by_pure_dict(state, jax.tree.map(jnp.zeros_like, pure_dict))
    assert isinstance(state, nnx.State)
    assert isinstance(state['kernel'], nnx.Variable)
    assert jnp.array_equal(state['kernel'][...], jnp.zeros((4, 5)))
    assert type(state['kernel']) == nnx.Param
    nnx.update(module, state)
    assert jnp.array_equal(module(jnp.ones((3, 4))), jnp.zeros((3, 5)))

  def test_diff(self):
    class MLPs(nnx.Module):
      def __init__(self, dim, rngs: nnx.Rngs, n=4):
        self.layers = nnx.List()
        for _ in range(n):
          self.layers.append(nnx.Linear(dim, dim, rngs=rngs, use_bias=False))

      def __call__(self, x):
        for layer in self.layers:
          x = layer(x)
        return x

    model1 = MLPs(4, rngs=nnx.Rngs(0), n=4)
    model2 = MLPs(4, rngs=nnx.Rngs(1), n=4)
    model3 = MLPs(4, rngs=nnx.Rngs(1), n=5)

    self.assertEqual(
      nnx.statelib.diff(nnx.state(model2), nnx.state(model1)),
      nnx.state({})
    )
    self.assertNotEqual(
      nnx.statelib.diff(nnx.state(model3), nnx.state(model1)),
      nnx.state({})
    )
    self.assertEqual(
      nnx.statelib.diff(nnx.state(model1), nnx.state(model3)),
      nnx.state({})
    )


if __name__ == '__main__':
  absltest.main()