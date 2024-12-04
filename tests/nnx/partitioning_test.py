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

from typing import TYPE_CHECKING
from absl.testing import absltest
from flax import nnx
import jax


class List(nnx.Module):
  def __init__(self, items):
    vars(self).update({str(i): item for i, item in enumerate(items)})

  def __getitem__(self, idx):
    return getattr(self, str(idx))

  def __setitem__(self, idx, value):
    setattr(self, str(idx), value)


class Dict(nnx.Module):
  def __init__(self, *args, **kwargs):
    vars(self).update(dict(*args, **kwargs))

  def __getitem__(self, key):
    return vars(self)[key]

  def __setitem__(self, key, value):
    vars(self)[key] = value

  if TYPE_CHECKING:

    def __getattr__(self, key): ...


class TestPartitioning(absltest.TestCase):

  def test_partition(self):
    m = Dict(
      a=List([nnx.Param(1), nnx.BatchStat(2)]),
      b=nnx.Param(2),
      c=100,
    )

    graphdef, params, rest = nnx.split(m, nnx.Param, ...)

    self.assertLen(params, 2)
    self.assertLen(rest, 1)

    # check params
    self.assertEqual(params['a']['0'].value, m.a['0'].value)
    self.assertEqual(params['b'].value, m.b.value)

    # check rest
    self.assertEqual(rest['a']['1'].value, m.a['1'].value)

    m2 = nnx.merge(graphdef, params, rest)

    self.assertEqual(m2.a['0'].value, m.a['0'].value)
    self.assertEqual(m2.a['1'].value, m.a['1'].value)
    self.assertEqual(m2.b.value, m.b.value)
    self.assertEqual(m2.c, 100)

  def test_complete_partitioning(self):
    m = Dict(
      a=List([nnx.Param(1), nnx.Param(2), nnx.Variable(3)]),
      b=Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
    )

    # no error
    nnx.split(m, nnx.Param, nnx.BatchStat, nnx.Variable)

  def test_complete_partitioning_plus_ellipsis(self):
    m = Dict(
      a=List([nnx.Param(1), nnx.Param(2), nnx.Variable(3)]),
      b=Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
    )

    # no error if additional ... is passed at the end
    nnx.split(m, nnx.Param, nnx.BatchStat, nnx.Variable, ...)

  def test_inclomplete_partition_error(self):
    m = Dict(
      a=List([nnx.Param(1), nnx.Param(2), nnx.Variable(3)]),
      b=Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
    )

    with self.assertRaisesRegex(
        ValueError, 'Non-exhaustive filters, got a non-empty remainder'
    ):
      nnx.split(m, nnx.Param)

  def test_ellipsis_not_last_error(self):
    m = Dict(
      a=List([nnx.Param(1), nnx.Param(2), nnx.Variable(3)]),
      b=Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
    )

    with self.assertRaisesRegex(
        ValueError, '`...` or `True` can only be used as the last filters'
    ):
      nnx.split(m, ..., nnx.Param)

  def test_update_from(self):
    m = Dict(
      a=List([nnx.Param(1), nnx.BatchStat(3)]),
      b=nnx.Param(2),
      c=100,
    )

    state = nnx.split(
      m,
    )[1]
    state = jax.tree.map(lambda x: x * 2, state)

    nnx.update(m, state)

    self.assertEqual(m.a[0].value, 2)
    self.assertEqual(m.a[1].value, 6)
    self.assertEqual(m.b.value, 4)
    self.assertEqual(m.c, 100)

  def test_update_from_with_array_leaf(self):
    m = Dict(
      a=List([nnx.Param(1), nnx.BatchStat(3)]),
      b=nnx.Param(2),
      c=nnx.Variable(jax.numpy.array(100)),
    )

    graphdef, state = nnx.split(
      m,
    )
    state = jax.tree.map(lambda x: x * 2, state)

    nnx.update(m, state)

    self.assertEqual(m.a[0].value, 2)
    self.assertEqual(m.a[1].value, 6)
    self.assertEqual(m.b.value, 4)
    self.assertEqual(m.c.value, 200)

  def test_grad_example(self):
    m = Dict(
      a=List([nnx.Param(1.0), nnx.BatchStat(-10)]),
      b=nnx.Param(2.0),
      c=100,
    )

    params = nnx.state(m, nnx.Param)

    def loss(params):
      return sum(2 * p for p in jax.tree_util.tree_leaves(params))

    grads = jax.grad(loss)(params)
    nnx.update(m, grads)

    self.assertEqual(m.a[0].value, 2.0)
    self.assertEqual(m.a[1].value, -10)
    self.assertEqual(m.b.value, 2.0)
    self.assertEqual(m.c, 100)

  def test_get_paritition(self):
    m = Dict(
      a=List([nnx.Param(10.0), nnx.Param(20.0)]),
      b=nnx.Param(10.0),
      c=7,
      d=5.0,
    )

    # test Variables not shared
    self.assertIsNot(vars(m.a)['0'], vars(m)['b'])

    state = nnx.state(m, nnx.Variable)
    self.assertEqual(state['a']['0'].value, m.a['0'].value)
    self.assertEqual(state['a']['1'].value, m.a['1'].value)
    self.assertEqual(state['b'].value, m.b.value)
    self.assertIsNot(state['b'], state['a']['0'])
    self.assertLen(nnx.to_flat_state(state), 3)


if __name__ == '__main__':
  absltest.main()
