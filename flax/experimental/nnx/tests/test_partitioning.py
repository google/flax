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


import jax
import pytest

from flax.experimental import nnx


class TestPartitioning:
  def test_partition(self):
    m = nnx.Dict(
      a=nnx.Sequence([nnx.Param(1), nnx.BatchStat(2)]),
      b=nnx.Param(2),
      c=100,
    )

    params, rest, graphdef = m.split(nnx.Param, ...)

    assert len(params) == 2
    assert len(rest) == 1

    # check params
    assert params['a']['0'].raw_value == m.a[0].value
    assert params['b'].raw_value == m.b.value

    # check rest
    assert rest['a']['1'].raw_value == m.a[1].value

    m2 = graphdef.merge(params, rest)

    assert m2.a[0].value == m.a[0].value
    assert m2.a[1].value == m.a[1].value
    assert m2.b.value == m.b.value
    assert m2.c == 100

  def test_complete_partitioning(self):
    m = nnx.Dict(
      a=nnx.Sequence([nnx.Param(1), nnx.Param(2), nnx.Variable(3)]),
      b=nnx.Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
    )

    # no error
    m.split(nnx.Param, nnx.BatchStat, nnx.Variable)

  def test_complete_partitioning_plus_ellipsis(self):
    m = nnx.Dict(
      a=nnx.Sequence([nnx.Param(1), nnx.Param(2), nnx.Variable(3)]),
      b=nnx.Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
    )

    # no error if additional ... is passed at the end
    m.split(nnx.Param, nnx.BatchStat, nnx.Variable, ...)

  def test_inclomplete_partition_error(self):
    m = nnx.Dict(
      a=nnx.Sequence([nnx.Param(1), nnx.Param(2), nnx.Variable(3)]),
      b=nnx.Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
    )

    with pytest.raises(
      ValueError, match='Non-exhaustive filters, got a non-empty remainder'
    ):
      m.split(nnx.Param)

  def test_ellipsis_not_last_error(self):
    m = nnx.Dict(
      a=nnx.Sequence([nnx.Param(1), nnx.Param(2), nnx.Variable(3)]),
      b=nnx.Dict(c=nnx.Param(1), d=nnx.BatchStat(2)),
    )

    with pytest.raises(
      ValueError, match='Ellipsis `...` can only be used as the last filter,'
    ):
      m.split(..., nnx.Param)

  def test_update_from(self):
    m = nnx.Dict(
      a=nnx.Sequence([nnx.Param(1), nnx.BatchStat(3)]),
      b=nnx.Param(2),
      c=100,
    )

    state = m.split()[0]
    state = jax.tree_map(lambda x: x * 2, state)

    m.update(state)

    assert m.a[0].value == 2
    assert m.a[1].value == 6
    assert m.b.value == 4
    assert m.c == 100

  def test_update_from_with_array_leaf(self):
    m = nnx.Dict(
      a=nnx.Sequence([nnx.Param(1), nnx.BatchStat(3)]),
      b=nnx.Param(2),
      c=nnx.Variable(jax.numpy.array(100)),
    )

    state, graphdef = m.split()
    state = jax.tree_map(lambda x: x * 2, state)

    m.update(state)

    assert m.a[0].value == 2
    assert m.a[1].value == 6
    assert m.b.value == 4
    assert m.c.value == 200

  def test_grad_example(self):
    m = nnx.Dict(
      a=nnx.Sequence([nnx.Param(1.0), nnx.BatchStat(-10)]),
      b=nnx.Param(2.0),
      c=100,
    )

    params = m.extract(nnx.Param)

    def loss(params):
      return sum(2 * p for p in jax.tree_util.tree_leaves(params))

    grads = jax.grad(loss)(params)
    m.update(grads)

    assert m.a[0].value == 2.0
    assert m.a[1].value == -10
    assert m.b.value == 2.0
    assert m.c == 100

  def test_get_paritition(self):
    m = nnx.Dict(
      a=nnx.Sequence([nnx.Param(10.0), nnx.Param(20.0)]),
      b=nnx.Param(10.0),
      c=7,
      d=5.0,
    )

    # test Variables not shared
    assert vars(m.a)['0'] is not vars(m)['b']

    state = m.extract(nnx.Variable)
    assert state['a']['0'].raw_value == m.a[0].value
    assert state['a']['1'].raw_value == m.a[1].value
    assert state['b'].raw_value == m.b.value
    assert state.b is not state.a[0]
    assert len(state.flat_state()) == 3
