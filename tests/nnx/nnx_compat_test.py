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

"""Tests for flax.nnx_compat module."""

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp


class CompatTest(absltest.TestCase):

  def test_split_with_shared_variables(self):
    """split should handle duplicate Variables (graph=True behavior)."""
    v = nnx.Param(1)
    g = [v, v]

    graphdef, state = nnx.compat.split(g)

    # graph=True deduplicates: only 1 leaf in state
    self.assertLen(nnx.to_flat_state(state), 1)

    g2 = nnx.merge(graphdef, state)
    self.assertIs(g2[0], g2[1])

  def test_split_shared_module(self):
    """split should preserve shared references in modules."""
    m1 = nnx.Dict(a=nnx.Param(1), b=nnx.Param(2))
    m2 = nnx.Dict(x=m1, y=m1, z=nnx.Param(3))

    graphdef, state = nnx.compat.split(m2)
    m3 = nnx.merge(graphdef, state)

    self.assertIs(m3['x'], m3['y'])
    self.assertIs(m3['x']['a'], m3['y']['a'])

  def test_split_self_referencing_module(self):
    """split should handle self-referencing modules (graph=True behavior)."""
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1)
        self.sub = self

    m = Foo()
    graphdef, state = nnx.compat.split(m)
    self.assertLen(nnx.to_flat_state(state), 1)

    m2 = nnx.merge(graphdef, state)
    self.assertIs(m2, m2.sub)

  def test_state_with_shared_variables(self):
    """state should handle shared variables (graph=True)."""
    class Foo(nnx.Module):
      def __init__(self):
        p = nnx.Param(jnp.array(1))
        self.a = p
        self.b = p

    m = Foo()
    s = nnx.compat.state(m)
    # With graph=True, shared variables are deduplicated
    self.assertIn('a', s)
    self.assertNotIn('b', s)

  def test_partial_overrides_allowed(self):
    """partial should allow overriding graph=False."""
    v = nnx.Param(1)
    g = [v, v]

    with self.assertRaisesRegex(
        ValueError, 'does not support shared references'
    ):
      nnx.compat.split(g, graph=False)


if __name__ == '__main__':
  absltest.main()
