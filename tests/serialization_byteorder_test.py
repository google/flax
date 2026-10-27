# Copyright 2026 The Flax Authors.
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

"""Regression tests for byte order in serialized NumPy parameter arrays."""

import jax.numpy as jnp
import msgpack
import numpy as np
import pytest

from flax import serialization
from flax.core import freeze


@pytest.mark.parametrize('dtype', ['i2', 'u4', 'f4', 'f8', 'c8'])
@pytest.mark.parametrize('shape', [(), (2,), (1, 2)])
def test_non_native_byteorder_checkpoint_roundtrip(dtype, shape):
  dtype = np.dtype(dtype).newbyteorder('S')
  values = np.array([1, 258], dtype=dtype)
  array = values[:1].reshape(shape) if shape == () else values.reshape(shape)
  original_bytes = array.tobytes()
  state = freeze({'params': {'kernel': array}})

  restored = serialization.from_bytes(state, serialization.to_bytes(state))
  kernel = restored['params']['kernel']

  np.testing.assert_array_equal(kernel, array)
  assert kernel.dtype == array.dtype
  assert kernel.shape == array.shape
  assert array.tobytes() == original_bytes


@pytest.mark.parametrize('dtype', ['i2', 'f4', 'c8'])
def test_non_native_byteorder_chunked_roundtrip(dtype, monkeypatch):
  array = np.array([1, 2, 258, 4], dtype=np.dtype(dtype).newbyteorder('S'))
  monkeypatch.setattr(serialization, 'MAX_CHUNK_SIZE', array.dtype.itemsize * 2)

  restored = serialization.msgpack_restore(serialization.msgpack_serialize(array))

  np.testing.assert_array_equal(restored, array)


@pytest.mark.parametrize('dtype', ['i2', 'f4', 'c8'])
def test_native_byteorder_wire_compatibility(dtype):
  array = np.array([1, 2], dtype=dtype)
  legacy = msgpack.ExtType(
    serialization._MsgpackExtType.ndarray,
    msgpack.packb((array.shape, array.dtype.name, array.tobytes()), use_bin_type=True),
  )
  legacy_bytes = msgpack.packb(legacy)

  assert serialization.msgpack_serialize(array) == legacy_bytes
  np.testing.assert_array_equal(serialization.msgpack_restore(legacy_bytes), array)


@pytest.mark.parametrize('dtype', [jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.int4])
@pytest.mark.parametrize('chunked', [False, True])
def test_non_native_custom_dtype_checkpoint_roundtrip(dtype, chunked, monkeypatch):
  dtype = np.dtype(dtype)
  array = np.array([1, 2], dtype=dtype)
  # Changing byte order metadata cannot change these one-byte numeric values.
  array = array.view(array.dtype.newbyteorder('S'))
  assert array.dtype.itemsize == 1
  assert not array.dtype.isnative
  original_bytes = array.tobytes()
  if chunked:
    monkeypatch.setattr(serialization, 'MAX_CHUNK_SIZE', 1)
  state = freeze({'params': {'kernel': array}})

  restored = serialization.from_bytes(state, serialization.to_bytes(state))
  kernel = restored['params']['kernel']

  assert kernel.dtype.type is dtype.type
  np.testing.assert_array_equal(kernel.astype(np.float32), [1, 2])
  np.testing.assert_array_equal(
    np.asarray(jnp.asarray(kernel)).astype(np.float32), [1, 2]
  )
  assert array.tobytes() == original_bytes
  assert array.dtype.type is dtype.type
  assert not array.dtype.isnative


@pytest.mark.parametrize(
  'dtype', [jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.int4]
)
@pytest.mark.parametrize('swap_metadata', [False, True])
def test_custom_dtype_wire_compatibility(dtype, swap_metadata):
  array = np.array([1, 2], dtype=dtype)
  if swap_metadata:
    array = array.view(array.dtype.newbyteorder('S'))
  legacy = msgpack.ExtType(
    serialization._MsgpackExtType.ndarray,
    msgpack.packb(
      (array.shape, array.dtype.name, array.tobytes()), use_bin_type=True
    ),
  )

  # Registered names retain dtype identity; str may describe void bytes or
  # even an unrecognized format such as '>f1'. Preserve their existing encoding.
  assert serialization.msgpack_serialize(array) == msgpack.packb(legacy)
