# Copyright 2022 The Flax Authors.
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

"""Tests for flax.struct and flax.serialization."""

import platform
import collections
from typing import NamedTuple, Any
import pytest

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax import serialization
from flax import struct
from flax.core import freeze
from flax.training import train_state
import jax
from jax import random
from jax.tree_util import Partial
import jax.numpy as jnp
import msgpack
import numpy as np
import optax


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


@struct.dataclass
class Point:
  x: float
  y: float
  meta: Any = struct.field(pytree_node=False)


class OriginalTuple(NamedTuple):
  value: Any


class WrongTuple(NamedTuple):
  wrong_field: Any


class OriginalModule(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(10)(x)
    return x


class WrongModule(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(10)(x)
    x = nn.Dense(10)(x)
    return x


class SerializationTest(parameterized.TestCase):

  def test_dataclass_serialization(self):
    p = Point(x=1, y=2, meta={'dummy': True})
    state_dict = serialization.to_state_dict(p)
    self.assertEqual(state_dict, {
        'x': 1,
        'y': 2,
    })
    restored_p = serialization.from_state_dict(p, {'x': 3, 'y': 4})
    expected_p = Point(x=3, y=4, meta={'dummy': True})
    self.assertEqual(restored_p, expected_p)

    with self.assertRaises(ValueError):  # invalid field
      serialization.from_state_dict(p, {'z': 3})
    with self.assertRaises(ValueError):  # missing field
      serialization.from_state_dict(p, {'x': 3})

  def test_model_serialization(self):
    rng = random.PRNGKey(0)
    module = nn.Dense(features=1, kernel_init=nn.initializers.ones_init())
    x = jnp.ones((1, 1), jnp.float32)
    initial_params = module.init(rng, x)
    state = serialization.to_state_dict(initial_params)
    self.assertEqual(state, {
        'params': {
            'kernel': np.ones((1, 1)),
            'bias': np.zeros((1,)),
        }
    })
    state = {
        'params': {
            'kernel': np.zeros((1, 1)),
            'bias': np.zeros((1,)),
        }
    }
    restored_model = serialization.from_state_dict(initial_params, state)
    self.assertEqual(restored_model, freeze(state))

  def test_partial_serialization(self):
    add_one = Partial(jnp.add, 1)
    state = serialization.to_state_dict(add_one)
    self.assertEqual(state, {
        'args': {'0': 1},
        'keywords': {}
    })
    restored_add_one = serialization.from_state_dict(add_one, state)
    self.assertEqual(add_one.args, restored_add_one.args)

  def test_optimizer_serialization(self):
    rng = random.PRNGKey(0)
    module = nn.Dense(features=1, kernel_init=nn.initializers.ones_init())
    x = jnp.ones((1, 1), jnp.float32)
    initial_params = module.init(rng, x)
    tx = optax.sgd(0.1, momentum=0.1)
    tx_state = tx.init(initial_params)
    state = serialization.to_state_dict(tx_state)
    expected_state = {
        '0': {
            'trace': {
                'params': {
                    'bias': np.array([0.], dtype=jnp.float32),
                    'kernel': np.array([[0.]], dtype=jnp.float32)
                    }
                }
            },
        '1': {}
        }
    self.assertEqual(state, expected_state)
    state = jax.tree_map(lambda x: x + 1, expected_state)
    restored_tx_state = serialization.from_state_dict(tx_state, state)
    tx_state_plus1 = jax.tree_map(lambda x: x + 1, tx_state)
    self.assertEqual(restored_tx_state, tx_state_plus1)

  def test_collection_serialization(self):

    @struct.dataclass
    class DummyDataClass:
      x: float

      @classmethod
      def initializer(cls, shape):
        del shape
        return cls(x=0.)

    class StatefulModule(nn.Module):
      @nn.compact
      def __call__(self):
        state = self.variable('state', 'dummy', DummyDataClass.initializer, ())
        state.value = state.value.replace(x=state.value.x + 1.)

    initial_variables = StatefulModule().init(random.PRNGKey(0))
    _, variables = StatefulModule().apply(initial_variables, mutable=['state'])
    serialized_state_dict = serialization.to_state_dict(variables)
    self.assertEqual(serialized_state_dict,
                     {'state': {'dummy': {'x': 2.0}}})
    deserialized_state = serialization.from_state_dict(variables,
                                                       serialized_state_dict)
    self.assertEqual(variables, deserialized_state)

  @parameterized.parameters(
    ['byte', 'b', 'ubyte', 'short',
    'h', 'ushort', 'i', 'uint', 'intp',
    'p', 'uintp', 'long', 'l', 'longlong',
    'q', 'ulonglong', 'half', 'e', 'f',
    'double', 'd', 'longdouble', 'g',
    'cfloat', 'cdouble', 'clongdouble', 'm',
    'bool8', 'b1', 'int64', 'i8', 'uint64', 'u8',
    'float16', 'f2', 'float32', 'f4', 'float64',
    'f8', 'float128', 'f16', 'complex64', 'c8',
    'complex128', 'c16', 'complex256', 'c32',
    'm8', 'int32', 'i4', 'uint32', 'u4', 'int16',
    'i2', 'uint16', 'u2', 'int8', 'i1', 'uint8',
    'u1', 'complex_', 'int0', 'uint0', 'single',
    'csingle', 'singlecomplex', 'float_', 'intc',
    'uintc', 'int_', 'longfloat', 'clongfloat',
    'longcomplex', 'bool_', 'int', 'float',
    'complex', 'bool']
  )
  def test_numpy_serialization(self, dtype):
    np.random.seed(0)
    if (dtype in {'float128', 'f16', 'complex256', 'c32'}) and (platform.system() == 'Darwin') and (platform.machine() == 'arm64'):
      pytest.skip(f'Mac M1 does not support dtype {dtype}') # skip testing these dtypes if user is on Mac M1

    v = np.random.uniform(-100, 100, size=()).astype(dtype)[()]
    restored_v = serialization.msgpack_restore(
        serialization.msgpack_serialize(v))
    self.assertEqual(restored_v.dtype, v.dtype)
    np.testing.assert_array_equal(restored_v, v)

    for shape in [(), (5,), (10, 10), (1, 20, 30, 1)]:
      arr = np.random.uniform(-100, 100, size=shape).astype(dtype)
      restored_arr = serialization.msgpack_restore(
          serialization.msgpack_serialize(arr))
      self.assertEqual(restored_arr.dtype, arr.dtype)
      np.testing.assert_array_equal(restored_arr, arr)

  def test_jax_numpy_serialization(self):
    jax_dtypes = [jnp.bool_, jnp.uint8, jnp.uint16, jnp.uint32,
                  jnp.int8, jnp.int16, jnp.int32,
                  jnp.bfloat16, jnp.float16, jnp.float32,
                  jnp.complex64]
    for dtype in jax_dtypes:
      v = jnp.array(np.random.uniform(-100, 100, size=())).astype(dtype)[()]
      restored_v = serialization.msgpack_restore(
          serialization.msgpack_serialize(v))
      self.assertEqual(restored_v.dtype, v.dtype)
      np.testing.assert_array_equal(restored_v, v)

      for shape in [(), (5,), (10, 10), (1, 20, 30, 1)]:
        arr = jnp.array(
            np.random.uniform(-100, 100, size=shape)).astype(dtype)
        restored_arr = serialization.msgpack_restore(
            serialization.msgpack_serialize(arr))
        self.assertEqual(restored_arr.dtype, arr.dtype)
        np.testing.assert_array_equal(restored_arr, arr)

  def test_complex_serialization(self):
    for x in [1j, 1+2j]:
      restored_x = serialization.msgpack_restore(
          serialization.msgpack_serialize(x))
      self.assertEqual(x, restored_x)

  def test_restore_chunked(self):
    old_chunksize = serialization.MAX_CHUNK_SIZE
    serialization.MAX_CHUNK_SIZE = 91 * 8
    try:
      tmp = np.random.uniform(-100, 100, size=(21, 37))
      serialized = serialization.to_bytes(tmp)
      restored = serialization.msgpack_restore(serialized)
    finally:
      serialization.MAX_CHUNK_SIZE = old_chunksize

    np.testing.assert_array_equal(restored, tmp)

  def test_restore_unchunked(self):
    """Check if mgspack_restore works for unchunked inputs."""
    def msgpack_serialize_legacy(pytree):
      """Old implementation that was not chunking."""
      return msgpack.packb(pytree, default=serialization._msgpack_ext_pack,
                           strict_types=True)

    tmp = np.random.uniform(-100, 100, size=(21, 37))
    serialized = msgpack_serialize_legacy(tmp)
    old_chunksize = serialization.MAX_CHUNK_SIZE
    serialization.MAX_CHUNK_SIZE = 91 * 8
    try:
      restored = serialization.msgpack_restore(serialized)
    finally:
      serialization.MAX_CHUNK_SIZE = old_chunksize

    np.testing.assert_array_equal(restored, tmp)

  def test_namedtuple_serialization(self):
    foo_class = collections.namedtuple('Foo', 'a b c')
    x1 = foo_class(a=1, b=2, c=3)
    x1_serialized = serialization.to_bytes(x1)
    x2 = foo_class(a=0, b=0, c=0)
    restored_x1 = serialization.from_bytes(x2, x1_serialized)
    self.assertEqual(type(x1), type(restored_x1))
    self.assertEqual(x1, restored_x1)

  def test_namedtuple_restore_legacy(self):
    foo_class = collections.namedtuple('Foo', 'a b c')
    x1 = foo_class(a=1, b=2, c=3)
    legacy_encoding = {
        'name': 'Foo',
        'fields': {
            '0': 'a',
            '1': 'b',
            '2': 'c'
        },
        'values': {
            '0': 1,
            '1': 2,
            '2': 3
        },
    }
    x2 = foo_class(a=0, b=0, c=0)
    restored_x1 = serialization.from_state_dict(x2, legacy_encoding)
    self.assertEqual(type(x1), type(restored_x1))
    self.assertEqual(x1, restored_x1)

  def test_model_serialization_to_bytes(self):
    rng = random.PRNGKey(0)
    module = nn.Dense(features=1, kernel_init=nn.initializers.ones_init())
    initial_params = module.init(rng, jnp.ones((1, 1), jnp.float32))
    serialized_bytes = serialization.to_bytes(initial_params)
    restored_params = serialization.from_bytes(initial_params, serialized_bytes)
    self.assertEqual(restored_params, initial_params)

  def test_optimizer_serialization_to_bytes(self):
    rng = random.PRNGKey(0)
    module = nn.Dense(features=1, kernel_init=nn.initializers.ones_init())
    initial_params = module.init(rng, jnp.ones((1, 1), jnp.float32))
    # model = nn.Model(module, initial_params)
    tx = optax.sgd(0.1, momentum=0.1)
    tx_state = tx.init(initial_params)
    serialized_bytes = serialization.to_bytes(tx_state)
    restored_tx_state = serialization.from_bytes(tx_state, serialized_bytes)
    self.assertEqual(restored_tx_state, tx_state)

  def test_serialization_chunking(self):
    old_chunksize = serialization.MAX_CHUNK_SIZE
    serialization.MAX_CHUNK_SIZE = 91 * 8
    try:
      tmp = {'a': np.ones((10, 10))}
      tmp = serialization._chunk_array_leaves_in_place(tmp)
    finally:
      serialization.MAX_CHUNK_SIZE = old_chunksize
    test = jax.tree_map(jnp.shape, tmp)
    ref = {
        'a': {
            '__msgpack_chunked_array__': (),
            'chunks': {
                '0': (91,),
                '1': (9,)
            },
            'shape': {
                '0': (),
                '1': ()
            }
        }
    }
    self.assertEqual(test, ref)

  def test_serialization_chunking2(self):
    old_chunksize = serialization.MAX_CHUNK_SIZE
    serialization.MAX_CHUNK_SIZE = 91 * 8
    try:
      tmp = {'a': np.ones((10, 10))}
      tmpbytes = serialization.to_bytes(tmp)
      newtmp = serialization.from_bytes(tmp, tmpbytes)
    finally:
      serialization.MAX_CHUNK_SIZE = old_chunksize
    jax.tree_map(np.testing.assert_array_equal, tmp, newtmp)

  def test_serialization_chunking3(self):
    old_chunksize = serialization.MAX_CHUNK_SIZE
    serialization.MAX_CHUNK_SIZE = 91 * 8
    try:
      tmp = {'a': np.ones((10, 10))}
      tmpbytes = serialization.msgpack_serialize(tmp)
      newtmp = serialization.msgpack_restore(tmpbytes)
    finally:
      serialization.MAX_CHUNK_SIZE = old_chunksize

    jax.tree_map(np.testing.assert_array_equal, tmp, newtmp)

  @parameterized.parameters(
      {'target': [[[1, 2, 3], [4, 5]]], 'wrong_target': [[[1, 2, 3], [4]]],
       'msg': ('The size of the list and the state dict do not match,'
               ' got 1 and 2 at path ./0/1')},
      {'target': (((1, 2, 3), (4, 5)),),
       'wrong_target': (((1, 2, 3), (4,)),),
       'msg': ('The size of the list and the state dict do not match,'
               ' got 1 and 2 at path ./0/1')},
      {'target': (((1, 2, 3), (OriginalTuple([4, 5]), 6)),),
       'wrong_target': (((1, 2, 3), (WrongTuple([4, 5]), 6)),),
       'msg': ("The field names of the state dict and the named tuple do "
               "not match, got {'value'} and {'wrong_field'} at path ./0/1/0")},
      {'target': {'a': {'b': {'c': [1, 2, 3], 'd': [4, 5]}}},
       'wrong_target': {'a': {'b': {'c': [1, 2, 3], 'd': [4]}}},
       'msg': ('The size of the list and the state dict do not match,'
               ' got 1 and 2 at path ./a/b/d')},
      {'target': {'a': {'b': {'c': [1, 2, 3], 'd': [4, 5]}}},
       'wrong_target': {'a': {'b': {'c': [1, 2, 3], 'e': [4, 5]}}},
       'msg': ("The target dict keys and state dict keys do not match, "
               "target dict contains keys {'e'} which are not present in state dict at path ./a/b")},
      {'target': 'original_params',
       'wrong_target': 'wrong_params',
       'msg': ("The target dict keys and state dict keys do not match, "
               "target dict contains keys {'Dense_1'} which are not present in state dict at path ./params")},
      {'target': 'original_train_state',
       'wrong_target': 'wrong_train_state',
       'msg': ("The target dict keys and state dict keys do not match, "
               "target dict contains keys {'Dense_1'} which are not present in state dict at path ./params/params")}
  )
  def test_serialization_errors(self, target, wrong_target, msg):
    if target == 'original_params':
      x = jnp.ones((1, 28, 28, 1))
      rng = jax.random.PRNGKey(1)
      original_module = OriginalModule()
      target = original_module.init(rng, x)
      wrong_module = WrongModule()
      wrong_target = wrong_module.init(rng, x)

    elif target == 'original_train_state':
      x = jnp.ones((1, 28, 28, 1))
      rng = jax.random.PRNGKey(1)
      original_module = OriginalModule()
      original_params = original_module.init(rng, x)
      wrong_module = WrongModule()
      wrong_params = wrong_module.init(rng, x)

      tx = optax.sgd(learning_rate=0.1, momentum=0.9)
      target = train_state.TrainState.create(
          apply_fn=original_module.apply, params=original_params, tx=tx)
      wrong_target = train_state.TrainState.create(
          apply_fn=wrong_module.apply, params=wrong_params, tx=tx)

    encoded_bytes = serialization.to_bytes(target)
    with self.assertRaisesWithLiteralMatch(ValueError, msg):
      serialization.from_bytes(wrong_target, encoded_bytes)


if __name__ == '__main__':
  absltest.main()
