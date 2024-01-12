# Copyright 2023 The Flax Authors.
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

from absl.testing.absltest import TestCase

from flax.experimental import nnx


class StateTest(TestCase):
  def test_create_state(self):
    state = nnx.State({'a': nnx.Param(1), 'b': {'c': nnx.Param(2)}})

    assert state['a'] == 1
    assert state['b']['c'] == 2

  def test_get_attr(self):
    state = nnx.State({'a': nnx.Param(1), 'b': {'c': nnx.Param(2)}})

    assert state.a == 1
    assert state.b.c == 2

  def test_set_attr(self):
    state = nnx.State({'a': nnx.Param(1), 'b': {'c': nnx.Param(2)}})

    state.a = 3
    state.b.c = 4

    assert state['a'] == 3
    assert state['b']['c'] == 4

  def test_set_attr_variables(self):
    state = nnx.State({'a': nnx.Param(1), 'b': {'c': nnx.Param(2)}})

    state.a = 3
    state.b.c = 4

    assert isinstance(state.variables.a, nnx.Param)
    assert state.variables.a.value == 3
    assert isinstance(state.b.variables.c, nnx.Param)
    assert state.b.variables.c.value == 4

  def test_integer_access(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.layers = [nnx.Linear(1, 2, rngs=rngs), nnx.Linear(2, 3, rngs=rngs)]

    module = Foo(rngs=nnx.Rngs(0))
    state = module.get_state()

    assert module.layers[0].kernel.shape == (1, 2)
    assert state.layers[0].kernel.shape == (1, 2)
    assert module.layers[1].kernel.shape == (2, 3)
    assert state.layers[1].kernel.shape == (2, 3)
