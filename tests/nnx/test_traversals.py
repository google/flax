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

"""Tests for flax.nnx.traversal."""
from absl.testing import absltest
from flax.core import freeze
from flax.nnx import traversals
import jax

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class TraversalTest(absltest.TestCase):
  def test_flatten_mapping(self):
    xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    flat_xs = traversals.flatten_mapping(xs)
    self.assertEqual(
      flat_xs,
      {
        ('foo',): 1,
        ('bar', 'a'): 2,
      },
    )
    flat_xs = traversals.flatten_mapping(freeze(xs))
    self.assertEqual(
      flat_xs,
      {
        ('foo',): 1,
        ('bar', 'a'): 2,
      },
    )
    flat_xs = traversals.flatten_mapping(xs, sep='/')
    self.assertEqual(
      flat_xs,
      {
        'foo': 1,
        'bar/a': 2,
      },
    )

  def test_unflatten_mapping(self):
    expected_xs = {'foo': 1, 'bar': {'a': 2}}
    xs = traversals.unflatten_mapping(
      {
        ('foo',): 1,
        ('bar', 'a'): 2,
      }
    )
    self.assertEqual(xs, expected_xs)
    xs = traversals.unflatten_mapping(
      {
        'foo': 1,
        'bar/a': 2,
      },
      sep='/',
    )
    self.assertEqual(xs, expected_xs)

  def test_flatten_mapping_keep_empty(self):
    ys = {'a': {}}
    xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    flat_ys = traversals.flatten_mapping(ys, keep_empty_nodes=True)
    flat_xs = traversals.flatten_mapping(xs, keep_empty_nodes=True)
    empty_node = flat_ys[('a',)]
    self.assertEqual(
      flat_xs,
      {
        ('foo',): 1,
        ('bar', 'a'): 2,
        ('bar', 'b'): empty_node,
      },
    )
    xs_restore = traversals.unflatten_mapping(flat_xs)
    self.assertEqual(xs, xs_restore)

  def test_flatten_mapping_is_leaf(self):
    xs = {'foo': {'c': 4}, 'bar': {'a': 2, 'b': {}}}
    flat_xs = traversals.flatten_mapping(
      xs, is_leaf=lambda k, x: len(k) == 1 and len(x) == 2
    )
    self.assertEqual(
      flat_xs,
      {
        ('foo', 'c'): 4,
        ('bar',): {'a': 2, 'b': {}},
      },
    )
    xs_restore = traversals.unflatten_mapping(flat_xs)
    self.assertEqual(xs, xs_restore)


if __name__ == '__main__':
  absltest.main()
