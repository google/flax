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
    state = nnx.State({'a': nnx.Param.state(1), 'b': {'c': nnx.Param.state(2)}})

    assert state['a'].value == 1
    assert state['b']['c'].value == 2


  def test_pure_dict(self):
    module = nnx.Linear(4, 5, rngs=nnx.Rngs(0))
    state = nnx.state(module)
    pure_dict = nnx.to_pure_dict(state)
    assert isinstance(pure_dict, dict)
    assert isinstance(pure_dict['kernel'], jax.Array)
    assert isinstance(pure_dict['bias'], jax.Array)
    nnx.replace_by_pure_dict(state, jax.tree.map(jnp.zeros_like, pure_dict))
    assert isinstance(state, nnx.State)
    assert isinstance(state['kernel'], nnx.VariableState)
    assert jnp.array_equal(state['kernel'].value, jnp.zeros((4, 5)))
    assert state['kernel'].type == nnx.Param
    nnx.update(module, state)
    assert jnp.array_equal(module(jnp.ones((3, 4))), jnp.zeros((3, 5)))


if __name__ == '__main__':
  absltest.main()