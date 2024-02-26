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

"""Tests for flax.struct."""


import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from absl.testing import absltest

import flax
import flax.linen as nn
from flax.core import freeze
from flax.cursor import AccessType, _traverse_tree, cursor
from flax.errors import CursorFindError, TraverseTreeError
from flax.training import train_state

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class GenericTuple(NamedTuple):
  x: Any
  y: Any = None
  z: Any = None


@dataclasses.dataclass
class GenericDataClass:
  x: Any
  y: Any = None
  z: Any = None


class CursorTest(absltest.TestCase):
  def test_repr(self):
    g = GenericTuple(1, 'a', (2, 'b'))
    c = cursor(
      {'a': {1: {(2, 3): 'z', 4: g, '6': (7, 8)}, 'b': [1, 2, 3]}, 'z': -1}
    )
    self.assertEqual(
      repr(c),
      """Cursor(
  _obj={'a': {1: {(2, 3): 'z', 4: GenericTuple(x=1, y='a', z=(2, 'b')), '6': (7, 8)}, 'b': [1, 2, 3]}, 'z': -1},
  _changes={}
)""",
    )

    # test overwriting
    c['z'] = -2
    c['z'] = -3
    c['a']['b'][1] = -2
    c['a']['b'] = None

    # test deep mutation
    c['a'][1][4].x = (2, 4, 6)
    c['a'][1][4].z[0] = flax.core.freeze({'a': 1, 'b': {'c': 2, 'd': 3}})

    self.assertEqual(
      repr(c),
      """Cursor(
  _obj={'a': {1: {(2, 3): 'z', 4: GenericTuple(x=1, y='a', z=(2, 'b')), '6': (7, 8)}, 'b': [1, 2, 3]}, 'z': -1},
  _changes={
    'z': Cursor(
           _obj=-3,
           _changes={}
         ),
    'a': Cursor(
           _obj={1: {(2, 3): 'z', 4: GenericTuple(x=1, y='a', z=(2, 'b')), '6': (7, 8)}, 'b': [1, 2, 3]},
           _changes={
             'b': Cursor(
                    _obj=None,
                    _changes={}
                  ),
             1: Cursor(
                  _obj={(2, 3): 'z', 4: GenericTuple(x=1, y='a', z=(2, 'b')), '6': (7, 8)},
                  _changes={
                    4: Cursor(
                         _obj=GenericTuple(x=1, y='a', z=(2, 'b')),
                         _changes={
                           'x': Cursor(
                                  _obj=(2, 4, 6),
                                  _changes={}
                                ),
                           'z': Cursor(
                                  _obj=(2, 'b'),
                                  _changes={
                                    0: Cursor(
                                         _obj=FrozenDict({
                                             a: 1,
                                             b: {
                                                 c: 2,
                                                 d: 3,
                                             },
                                         }),
                                         _changes={}
                                       )
                                  }
                                )
                         }
                       )
                  }
                )
           }
         )
  }
)""",
    )

  def test_magic_methods(self):
    def same_value(v1, v2):
      if isinstance(v1, tuple):
        return all([
            jnp.all(jax.tree_util.tree_map(lambda x, y: x == y, e1, e2))
            for e1, e2 in zip(v1, v2)
        ])
      return jnp.all(jax.tree_util.tree_map(lambda x, y: x == y, v1, v2))

    list_obj = [(1, 2), (3, 4)]
    for l, tuple_wrap in ((list_obj, lambda x: x), (tuple(list_obj), tuple)):
      c = cursor(l)
      # test __len__
      self.assertTrue(same_value(len(c), len(l)))
      # test __iter__
      for i, child_c in enumerate(c):
        child_c[1] += i + 1
      self.assertEqual(c.build(), tuple_wrap([(1, 3), (3, 6)]))
      # test __reversed__
      for i, child_c in enumerate(reversed(c)):
        child_c[1] += i + 1
      self.assertEqual(c.build(), tuple_wrap([(1, 5), (3, 7)]))
    # test __iter__ error
    with self.assertRaisesRegex(
      NotImplementedError,
      '__iter__ method only implemented for tuples and lists, not type <class'
      " 'dict'>",
    ):
      c = cursor({'a': 1, 'b': 2})
      for key in c:
        c[key] *= -1
    # test __iter__ error
    with self.assertRaisesRegex(
      NotImplementedError,
      '__reversed__ method only implemented for tuples and lists, not type'
      " <class 'dict'>",
    ):
      c = cursor({'a': 1, 'b': 2})
      for key in reversed(c):
        c[key] *= -1

    for obj_value in (2, jnp.array([[1, -2], [3, 4]])):
      for c in (
        cursor(obj_value),
        cursor([obj_value])[0],
        cursor((obj_value,))[0],
        cursor({0: obj_value})[0],
        cursor(flax.core.freeze({0: obj_value}))[0],
        cursor(GenericTuple(x=obj_value)).x,
        cursor(GenericDataClass(x=obj_value)).x,
      ):
        # test __neg__
        self.assertTrue(same_value(-c, -obj_value))
        # test __pos__
        self.assertTrue(same_value(+c, +obj_value))
        # test __abs__
        self.assertTrue(same_value(abs(-c), abs(-obj_value)))
        # test __invert__
        self.assertTrue(same_value(~c, ~obj_value))
        # test __round__
        self.assertTrue(same_value(round(c + 0.123), round(obj_value + 0.123)))
        self.assertTrue(
          same_value(round(c + 0.123, 2), round(obj_value + 0.123, 2))
        )

        for other_value in (3, jnp.array([[5, 6], [7, 8]])):
          # test __add__
          self.assertTrue(same_value(c + other_value, obj_value + other_value))
          # test __radd__
          self.assertTrue(same_value(other_value + c, other_value + obj_value))
          # test __sub__
          self.assertTrue(same_value(c - other_value, obj_value - other_value))
          # test __rsub__
          self.assertTrue(same_value(other_value - c, other_value - obj_value))
          # test __mul__
          self.assertTrue(same_value(c * other_value, obj_value * other_value))
          # test __rmul__
          self.assertTrue(same_value(other_value * c, other_value * obj_value))
          # test __truediv__
          self.assertTrue(same_value(c / other_value, obj_value / other_value))
          # test __rtruediv__
          self.assertTrue(same_value(other_value / c, other_value / obj_value))
          # test __floordiv__
          self.assertTrue(
            same_value(c // other_value, obj_value // other_value)
          )
          # test __rfloordiv__
          self.assertTrue(
            same_value(other_value // c, other_value // obj_value)
          )
          # test __mod__
          self.assertTrue(same_value(c % other_value, obj_value % other_value))
          # test __rmod__
          self.assertTrue(same_value(other_value % c, other_value % obj_value))
          # test __divmod__
          self.assertTrue(
            same_value(divmod(c, other_value), divmod(obj_value, other_value))
          )
          # test __rdivmod__
          self.assertTrue(
            same_value(divmod(other_value, c), divmod(other_value, obj_value))
          )
          # test __pow__
          self.assertTrue(
            same_value(pow(c, other_value), pow(obj_value, other_value))
          )
          # test __rpow__
          self.assertTrue(
            same_value(pow(other_value, c), pow(other_value, obj_value))
          )
          # test __lshift__
          self.assertTrue(
            same_value(c << other_value, obj_value << other_value)
          )
          # test __rlshift__
          self.assertTrue(
            same_value(other_value << c, other_value << obj_value)
          )
          # test __rshift__
          self.assertTrue(
            same_value(c >> other_value, obj_value >> other_value)
          )
          # test __rrshift__
          self.assertTrue(
            same_value(other_value >> c, other_value >> obj_value)
          )
          # test __and__
          self.assertTrue(same_value(c & other_value, obj_value & other_value))
          # test __rand__
          self.assertTrue(same_value(other_value & c, other_value & obj_value))
          # test __xor__
          self.assertTrue(same_value(c ^ other_value, obj_value ^ other_value))
          # test __rxor__
          self.assertTrue(same_value(other_value ^ c, other_value ^ obj_value))
          # test __or__
          self.assertTrue(same_value(c | other_value, obj_value | other_value))
          # test __ror__
          self.assertTrue(same_value(other_value | c, other_value | obj_value))

          if isinstance(obj_value, jax.Array) and isinstance(
            other_value, jax.Array
          ):
            # test __matmul__
            self.assertTrue(
              same_value(c @ other_value, obj_value @ other_value)
            )
            # test __rmatmul__
            self.assertTrue(
              same_value(other_value @ c, other_value @ obj_value)
            )

          # test __lt__
          self.assertTrue(same_value(c < other_value, obj_value < other_value))
          self.assertTrue(same_value(other_value < c, other_value < obj_value))
          # test __le__
          self.assertTrue(
            same_value(c <= other_value, obj_value <= other_value)
          )
          self.assertTrue(
            same_value(other_value <= c, other_value <= obj_value)
          )
          # test __eq__
          self.assertTrue(
            same_value(c == other_value, obj_value == other_value)
          )
          self.assertTrue(
            same_value(other_value == c, other_value == obj_value)
          )
          # test __ne__
          self.assertTrue(
            same_value(c != other_value, obj_value != other_value)
          )
          self.assertTrue(
            same_value(other_value != c, other_value != obj_value)
          )
          # test __gt__
          self.assertTrue(same_value(c > other_value, obj_value > other_value))
          self.assertTrue(same_value(other_value > c, other_value > obj_value))
          # test __ge__
          self.assertTrue(
            same_value(c >= other_value, obj_value >= other_value)
          )
          self.assertTrue(
            same_value(other_value >= c, other_value >= obj_value)
          )

  def test_path(self):
    c = cursor(
      GenericTuple(
        x=[
          0,
          {'a': 1, 'b': (2, 3), ('c', 'd'): [4, 5]},
          (100, 200),
          [3, 4, 5],
        ],
        y=train_state.TrainState.create(
          apply_fn=lambda x: x,
          params=freeze({'a': 1, 'b': (2, 3), 'c': [4, 5]}),
          tx=optax.adam(1e-3),
        ),
      )
    )
    self.assertEqual(c.x[1][('c', 'd')][0]._path, ".x[1][('c', 'd')][0]")
    self.assertEqual(c.x[2][1]._path, '.x[2][1]')
    self.assertEqual(c.y.params['b'][1]._path, ".y.params['b'][1]")

    # test path when first access type is item access
    c = cursor([1, GenericTuple('a', 2), (3, 4)])
    self.assertEqual(c[1].x._path, '[1].x')
    self.assertEqual(c[2][0]._path, '[2][0]')

  def test_traverse_tree(self):
    c = cursor(
      GenericTuple(
        x=[
          0,
          {'a': 1, 'b': (2, 3), ('c', 'd'): [4, 5]},
          (100, 200),
          [3, 4, 5],
        ],
        y=3,
      )
    )

    def update_fn(path, value):
      if value == 4:
        return -4
      return value

    def cond_fn(path, value):
      return value == 3

    with self.assertRaisesRegex(
      TraverseTreeError,
      'Both update_fn and cond_fn are None. Exactly one of them must be'
      ' None.',
    ):
      next(_traverse_tree((), c._obj))
    with self.assertRaisesRegex(
      TraverseTreeError,
      'Both update_fn and cond_fn are not None. Exactly one of them must be'
      ' not None.',
    ):
      next(_traverse_tree((), c._obj, update_fn=update_fn, cond_fn=cond_fn))

    (p, v), (p2, v2) = _traverse_tree((), c._obj, update_fn=update_fn)
    self.assertEqual(
      p,
      (
        ('x', AccessType.ATTR),
        (1, AccessType.ITEM),
        (('c', 'd'), AccessType.ITEM),
        (0, AccessType.ITEM),
      ),
    )
    self.assertEqual(v, -4)
    self.assertEqual(
      p2, (('x', AccessType.ATTR), (3, AccessType.ITEM), (1, AccessType.ITEM))
    )
    self.assertEqual(v2, -4)

    p, p2, p3 = _traverse_tree((), c._obj, cond_fn=cond_fn)
    self.assertEqual(
      p,
      (
        ('x', AccessType.ATTR),
        (1, AccessType.ITEM),
        ('b', AccessType.ITEM),
        (1, AccessType.ITEM),
      ),
    )
    self.assertEqual(
      p2, (('x', AccessType.ATTR), (3, AccessType.ITEM), (0, AccessType.ITEM))
    )
    self.assertEqual(p3, (('y', AccessType.ATTR),))

  def test_set_and_build(self):
    # test regular dict and FrozenDict
    dict_obj = {'a': 1, 'b': (2, 3), 'c': [4, 5]}
    for d, freeze_wrap in ((dict_obj, lambda x: x), (freeze(dict_obj), freeze)):
      # set API
      self.assertEqual(
        cursor(d)['b'][0].set(10),
        freeze_wrap({'a': 1, 'b': (10, 3), 'c': [4, 5]}),
      )
      # build API
      c = cursor(d)
      c['b'][0] = 20
      c['a'] = (100, 200)
      d2 = c.build()
      self.assertEqual(
        d2, freeze_wrap({'a': (100, 200), 'b': (20, 3), 'c': [4, 5]})
      )
    self.assertEqual(
      dict_obj, {'a': 1, 'b': (2, 3), 'c': [4, 5]}
    )  # make sure original object is unchanged

    # test list and tuple
    list_obj = [0, dict_obj, (1, 2), [3, 4, 5]]
    for l, tuple_wrap in ((list_obj, lambda x: x), (tuple(list_obj), tuple)):
      # set API
      self.assertEqual(
        cursor(l)[1]['b'][0].set(10),
        tuple_wrap([0, {'a': 1, 'b': (10, 3), 'c': [4, 5]}, (1, 2), [3, 4, 5]]),
      )
      # build API
      c = cursor(l)
      c[1]['b'][0] = 20
      c[2] = (100, 200)
      l2 = c.build()
      self.assertEqual(
        l2,
        tuple_wrap(
          [0, {'a': 1, 'b': (20, 3), 'c': [4, 5]}, (100, 200), [3, 4, 5]]
        ),
      )
    self.assertEqual(
      list_obj, [0, {'a': 1, 'b': (2, 3), 'c': [4, 5]}, (1, 2), [3, 4, 5]]
    )  # make sure original object is unchanged

    # test TrainState
    state = train_state.TrainState.create(
      apply_fn=lambda x: x,
      params=dict_obj,
      tx=optax.adam(1e-3),
    )
    # set API
    self.assertEqual(
      cursor(state).params['b'][0].set(10).params,
      {'a': 1, 'b': (10, 3), 'c': [4, 5]},
    )
    # build API
    new_fn = lambda x: x + 1
    c = cursor(state)
    c.apply_fn = new_fn
    c.params['b'][0] = 20
    c.params['a'] = (100, 200)
    state2 = c.build()
    self.assertEqual(state2.apply_fn, new_fn)
    self.assertEqual(
      state2.params, {'a': (100, 200), 'b': (20, 3), 'c': [4, 5]}
    )

    self.assertEqual(
      dict_obj, {'a': 1, 'b': (2, 3), 'c': [4, 5]}
    )  # make sure original object is unchanged

    # test NamedTuple
    # set API
    t = GenericTuple(GenericTuple(0))
    self.assertEqual(cursor(t).x.x.set(1), GenericTuple(GenericTuple(1)))

    # build API
    c = cursor(t)
    c.x.x = 2
    c.x.y = 3
    c.y = 4
    t2 = c.build()
    self.assertEqual(t2, GenericTuple(GenericTuple(2, 3), 4))

    self.assertEqual(
      t, GenericTuple(GenericTuple(0))
    )  # make sure original object is unchanged

  def test_apply_update(self):
    # test list and tuple
    def update_fn(path, value):
      """Multiply the first element of all leaf nodes of the pytree by -1."""
      if path[-1] == '0' and isinstance(value, int):
        return value * -1
      return value

    for tuple_wrap in (lambda x: x, tuple):
      l = tuple_wrap([tuple_wrap([1, 2]), tuple_wrap([3, 4])])
      c = cursor(l)
      l2 = c.apply_update(update_fn).build()
      self.assertEqual(
        l2, tuple_wrap([tuple_wrap([-1, 2]), tuple_wrap([-3, 4])])
      )
      self.assertEqual(
        l, tuple_wrap([tuple_wrap([1, 2]), tuple_wrap([3, 4])])
      )  # make sure the original object is unchanged

    # test regular dict and FrozenDict
    def update_fn(path, value):
      """Multiply all dense kernel params by 2 and add 1.
      Subtract the Dense_1 bias param by 1."""
      if 'kernel' in path:
        return value * 2 + 1
      elif 'Dense_1' in path and 'bias' in path:
        return value - 1
      return value

    class Model(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.Dense(3)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)
        x = nn.relu(x)
        return x

    for freeze_wrap in (lambda x: x, freeze):
      params = freeze_wrap(
        Model().init(jax.random.key(0), jnp.empty((1, 2)))['params']
      )

      c = cursor(params)
      params2 = c.apply_update(update_fn).build()
      for layer in ('Dense_0', 'Dense_1', 'Dense_2'):
        self.assertTrue(
          (params2[layer]['kernel'] == 2 * params[layer]['kernel'] + 1).all()
        )
        if layer == 'Dense_1':
          self.assertTrue(
            (params2[layer]['bias'] == jnp.array([-1, -1, -1])).all()
          )
        else:
          self.assertTrue(
            (params2[layer]['bias'] == params[layer]['bias']).all()
          )
      self.assertTrue(
        jax.tree_util.tree_all(
          jax.tree_util.tree_map(
            lambda x, y: (x == y).all(),
            params,
            freeze_wrap(
              Model().init(jax.random.key(0), jnp.empty((1, 2)))['params']
            ),
          )
        )
      )  # make sure original params are unchanged

    # test TrainState
    def update_fn(path, value):
      """Replace params with empty dictionary."""
      if 'params' in path:
        return {}
      return value

    state = train_state.TrainState.create(
      apply_fn=lambda x: x,
      params={'a': 1, 'b': 2},
      tx=optax.adam(1e-3),
    )
    c = cursor(state)
    state2 = c.apply_update(update_fn).build()
    self.assertEqual(state2.params, {})
    self.assertEqual(
      state.params, {'a': 1, 'b': 2}
    )  # make sure original params are unchanged

    # test NamedTuple
    def update_fn(path, value):
      """Add 5 to all x-attribute values that are ints."""
      if path[-1] == 'x' and isinstance(value, int):
        return value + 5
      return value

    t = GenericTuple(GenericTuple(0, 1), GenericTuple(2, 3))
    c = cursor(t)
    t2 = c.apply_update(update_fn).build()
    self.assertEqual(t2, GenericTuple(GenericTuple(5, 1), GenericTuple(7, 3)))
    self.assertEqual(
      t, GenericTuple(GenericTuple(0, 1), GenericTuple(2, 3))
    )  # make sure original object is unchanged

  def test_apply_update_root_node_unmodified(self):
    def update_fn(path, value):
      if isinstance(value, list):
        value = value.copy()
        value.append(-1)
      return value

    l = [[1, 2], [3, 4], 5]
    l2 = cursor(l).apply_update(update_fn).build()
    self.assertEqual(l2, [[1, 2, -1], [3, 4, -1], 5])

  def test_multi_modify(self):
    d = {'a': 1, 'b': (2, 3), 'c': [4, 5]}
    c = cursor(d)
    # test multiple changes on same element
    c['b'][0] = 6
    c['b'][0] = 7
    d2 = c.build()
    self.assertEqual(d2, {'a': 1, 'b': (7, 3), 'c': [4, 5]})
    # test nested changes
    c['a'] = (100, 200)
    c['a'][1] = -1
    d3 = c.build()
    self.assertEqual(d3, {'a': (100, -1), 'b': (7, 3), 'c': [4, 5]})

  def test_hidden_change(self):
    # test list
    l = [1, 2]
    c = cursor(l)
    c[0] = 100
    l[1] = -1
    l2 = c.build()
    self.assertEqual(
      l2, [100, -1]
    )  # change in l affects l2 (this is expected behavior)
    self.assertEqual(l, [1, -1])

    # test regular dict
    d = {'a': 1, 'b': 2}
    c = cursor(d)
    c['a'] = 100
    d['b'] = -1
    d2 = c.build()
    self.assertEqual(
      d2, {'a': 100, 'b': -1}
    )  # change in d affects d2 (this is expected behavior)
    self.assertEqual(d, {'a': 1, 'b': -1})

    # test TrainState
    params = {'a': 1, 'b': 2}
    state = train_state.TrainState.create(
      apply_fn=lambda x: x,
      params=params,
      tx=optax.adam(1e-3),
    )
    c = cursor(state)
    c.params['a'] = 100
    params['b'] = -1
    state2 = c.build()
    self.assertEqual(
      state2.params, {'a': 100, 'b': -1}
    )  # change in state affects state2 (this is expected behavior)
    self.assertEqual(state.params, {'a': 1, 'b': -1})

  def test_named_tuple_multi_access(self):
    t = GenericTuple(GenericTuple(0, 1), GenericTuple(2, 3))
    c = cursor(t)
    c.x.x = 4
    c[0].y = 5
    c.y.x = 6
    c.y[1] = 7
    self.assertEqual(
      c.build(), GenericTuple(GenericTuple(4, 5), GenericTuple(6, 7))
    )

    c[0][1] = -5
    self.assertEqual(
      c.build(), GenericTuple(GenericTuple(4, -5), GenericTuple(6, 7))
    )
    c.x[1] = -6
    self.assertEqual(
      c.build(), GenericTuple(GenericTuple(4, -6), GenericTuple(6, 7))
    )
    c.x.y = -7
    self.assertEqual(
      c.build(), GenericTuple(GenericTuple(4, -7), GenericTuple(6, 7))
    )
    c[0].y = -8
    self.assertEqual(
      c.build(), GenericTuple(GenericTuple(4, -8), GenericTuple(6, 7))
    )

  def test_find(self):
    c = cursor(
      GenericTuple(
        x=[
          0,
          {'a': 1, 'b': (2, 3), ('c', 'd'): [4, 5]},
          (100, 200),
          [3, 4, 5],
        ],
        y=train_state.TrainState.create(
          apply_fn=lambda x: x,
          params=freeze({'a': 1, 'b': (2, 3), 'c': [4, 5]}),
          tx=optax.adam(1e-3),
        ),
      )
    )

    with self.assertRaisesRegex(
      CursorFindError,
      'More than one object found given the conditions of the cond_fn\\. '
      'The first two objects found have the following paths: '
      "\\.x\\[1]\\['b'] and \\.y\\.params\\['b'] ",
    ):
      c.find(lambda path, value: 'b' in path and isinstance(value, tuple))
    with self.assertRaisesRegex(
      CursorFindError,
      'No object found given the conditions of the cond_fn\\.',
    ):
      c.find(lambda path, value: 'b' in path and isinstance(value, str))

    self.assertEqual(
      c.find(lambda path, value: path.endswith('params/b'))[1].set(30).y.params,
      freeze({'a': 1, 'b': (2, 30), 'c': [4, 5]}),
    )

  def test_find_all(self):
    # test list and tuple
    def cond_fn(path, value):
      """Get all lists that are not the first element in its parent."""
      return path[-1] != '0' and isinstance(value, (tuple, list))

    for tuple_wrap in (lambda x: x, tuple):
      l = tuple_wrap(
        [tuple_wrap([1, 2]), tuple_wrap([3, 4]), tuple_wrap([5, 6])]
      )
      c = cursor(l)
      c2, c3 = c.find_all(cond_fn)
      c2[0] *= -1
      c3[1] *= -2
      self.assertEqual(
        c.build(),
        tuple_wrap(
          [tuple_wrap([1, 2]), tuple_wrap([-3, 4]), tuple_wrap([5, -12])]
        ),
      )
      self.assertEqual(
        l,
        tuple_wrap(
          [tuple_wrap([1, 2]), tuple_wrap([3, 4]), tuple_wrap([5, 6])]
        ),
      )  # make sure the original object is unchanged

    # test regular dict and FrozenDict
    def cond_fn(path, value):
      """Get the second and third dense params."""
      return 'Dense_1' in path or 'Dense_2' in path

    class Model(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.Dense(3)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)
        x = nn.relu(x)
        return x

    for freeze_wrap in (lambda x: x, freeze):
      params = freeze_wrap(
        Model().init(jax.random.PRNGKey(0), jnp.empty((1, 2)))['params']
      )
      c = cursor(params)
      for i, c2 in enumerate(c.find_all(cond_fn)):
        self.assertEqual(
          c2['kernel'].set(123)[f'Dense_{i+1}'],
          freeze_wrap({'kernel': 123, 'bias': params[f'Dense_{i+1}']['bias']}),
        )
      self.assertTrue(
        jax.tree_util.tree_all(
          jax.tree_util.tree_map(
            lambda x, y: (x == y).all(),
            params,
            freeze_wrap(
              Model().init(jax.random.PRNGKey(0), jnp.empty((1, 2)))['params']
            ),
          )
        )
      )  # make sure original params are unchanged

    # test TrainState
    def cond_fn(path, value):
      """Find TrainState params."""
      return 'params' in path

    state = train_state.TrainState.create(
      apply_fn=lambda x: x,
      params={'a': 1, 'b': 2},
      tx=optax.adam(1e-3),
    )
    c = cursor(state)
    c2 = list(c.find_all(cond_fn))
    self.assertEqual(len(c2), 1)
    c2 = c2[0]
    self.assertEqual(c2['b'].set(-1).params, {'a': 1, 'b': -1})
    self.assertEqual(
      state.params, {'a': 1, 'b': 2}
    )  # make sure original params are unchanged

    # test NamedTuple
    def cond_fn(path, value):
      """Get all GenericTuples that have int x-attribute values."""
      return isinstance(value, GenericTuple) and isinstance(value.x, int)

    t = GenericTuple(
      GenericTuple(0, 'a'), GenericTuple(1, 'b'), GenericTuple('c', 2)
    )
    c = cursor(t)
    c2, c3 = c.find_all(cond_fn)
    c2.x += 5
    c3.x += 6
    self.assertEqual(
      c.build(),
      GenericTuple(
        GenericTuple(5, 'a'), GenericTuple(7, 'b'), GenericTuple('c', 2)
      ),
    )
    self.assertEqual(
      t,
      GenericTuple(
        GenericTuple(0, 'a'), GenericTuple(1, 'b'), GenericTuple('c', 2)
      ),
    )  # make sure original object is unchanged


if __name__ == '__main__':
  absltest.main()
