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

from absl.testing.absltest import TestCase

from flax.experimental import nnx


class StateTest(TestCase):
  def test_create_state(self):
    state = nnx.State({'a': nnx.Param.state(1), 'b': {'c': nnx.Param.state(2)}})

    assert state['a'].value == 1
    assert state['b']['c'].value == 2

  def test_get_attr(self):
    state = nnx.State({'a': nnx.Param.state(1), 'b': {'c': nnx.Param.state(2)}})

    assert state.a.value == 1
    assert state.b.c.value == 2

  def test_set_attr(self):
    state = nnx.State({'a': nnx.Param.state(1), 'b': {'c': nnx.Param.state(2)}})

    state.a.value = 3
    state.b.c.value = 4

    assert state['a'].value == 3
    assert state['b']['c'].value == 4

  def test_set_attr_variables(self):
    state = nnx.State({'a': nnx.Param.state(1), 'b': {'c': nnx.Param.state(2)}})

    state.a.value = 3
    state.b.c.value = 4

    assert issubclass(state.a.type, nnx.Param)
    assert state.a.value == 3
    assert issubclass(state.b.c.type, nnx.Param)
    assert state.b.c.value == 4

  def test_integer_access(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.layers = [nnx.Linear(1, 2, rngs=rngs), nnx.Linear(2, 3, rngs=rngs)]

    module = Foo(rngs=nnx.Rngs(0))
    state = nnx.state(module)

    assert module.layers[0].kernel.value.shape == (1, 2)
    assert state.layers[0].kernel.value.shape == (1, 2)
    assert module.layers[1].kernel.value.shape == (2, 3)
    assert state.layers[1].kernel.value.shape == (2, 3)
