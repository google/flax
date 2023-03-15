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

"""Tests for flax.training.checkpoints."""

import copy
import os
import pathlib
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from flax import config
from flax import core
from flax import errors
from flax import io
from flax import linen as nn
from flax import struct
from flax.training import checkpoints
import jax
from jax import numpy as jnp
import numpy as np

import orbax.checkpoint as orbax

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


PyTree = Any


def check_eq(xs, ys):
  return jax.tree_util.tree_all(
      jax.tree_util.tree_map(np.testing.assert_allclose, xs, ys))


def shuffle(l):
  """Functional shuffle."""
  l = copy.copy(l)
  np.random.shuffle(l)
  return l


class Inner(nn.Module):
  """Inner class based on nn."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(10, (2, 2))(x)
    x = nn.normalization.BatchNorm(True)(x)
    return x


class Model(nn.Module):
  """Simple model based on nn."""

  @nn.compact
  def __call__(self, inputs):
    x = nn.Conv(10, (2, 2))(inputs)
    x = Inner()(x)
    x = x.reshape([x.shape[0], -1])
    x = nn.normalization.BatchNorm(True)(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x


@struct.dataclass
class CustomDC:
  foo: Any
  bar: Any


class CheckpointsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config.update('flax_use_orbax_checkpointing', False)  # default value

  def test_naturalsort(self):
    np.random.seed(0)
    tests = [
        ['file_1', 'file_2', 'file_10', 'file_11', 'file_21'],
        ['file_0.001', 'file_0.01', 'file_0.1', 'file_1'],
        ['file_-3.0', 'file_-2', 'file_-1', 'file_0.0'],
        ['file_1e1', 'file_1.0e2', 'file_1e3', 'file_1.0e4'],
        ['file_1', 'file_2', 'file_9', 'file_1.0e1', 'file_11'],
    ]
    for test in tests:
      self.assertEqual(test, checkpoints.natural_sort(shuffle(test)))

  def test_safe_normpath(self):
    tests = ['./a/b/c', '/a//b/c', '/a/../b/c', 'a/b/./c', 'gs://a//b/c']
    expected = ['a/b/c', '/a/b/c', '/b/c', 'a/b/c', 'gs://a/b/c']
    for test, expect in zip(tests, expected):
      self.assertEqual(expect, checkpoints.safe_normpath(test))

  @parameterized.parameters({'use_orbax': True}, {'use_orbax': False})
  def test_save_restore_checkpoints(self, use_orbax):
    config.update('flax_use_orbax_checkpointing', use_orbax)
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    test_object0 = {'a': np.array([0, 0, 0], np.int32),
                    'b': np.array([0, 0, 0], np.int32)}
    test_object1 = {'a': np.array([1, 2, 3], np.int32),
                    'b': np.array([1, 1, 1], np.int32)}
    test_object2 = {'a': np.array([4, 5, 6], np.int32),
                    'b': np.array([2, 2, 2], np.int32)}
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, prefix='test_')
    check_eq(new_object, test_object0)
    # Create leftover temporary checkpoint, which should be ignored.
    io.GFile(os.path.join(tmp_dir, 'test_tmp'), 'w')
    checkpoints.save_checkpoint(
        tmp_dir, test_object1, 0, prefix='test_', keep=1)
    self.assertIn('test_0', os.listdir(tmp_dir))
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, prefix='test_')
    check_eq(new_object, test_object1)
    checkpoints.save_checkpoint(
        tmp_dir, test_object1, 1, prefix='test_', keep=1)
    checkpoints.save_checkpoint(
        tmp_dir, test_object2, 2, prefix='test_', keep=1)
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, prefix='test_')
    check_eq(new_object, test_object2)
    checkpoints.save_checkpoint(
        tmp_dir, test_object2, 3, prefix='test_', keep=2)
    checkpoints.save_checkpoint(
        tmp_dir, test_object1, 4, prefix='test_', keep=2)
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, prefix='test_')
    check_eq(new_object, test_object1)
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, step=3, prefix='test_')
    check_eq(new_object, test_object2)

    # Restore with a specific checkpoint path, not the directory path.
    new_object = checkpoints.restore_checkpoint(
        os.path.join(tmp_dir, 'test_3'), test_object0)
    check_eq(new_object, test_object2)
    # If a specific path is specified, but it does not exist, the same behavior
    # as when a directory is empty should apply: the target is returned
    # unchanged.
    new_object = checkpoints.restore_checkpoint(
        os.path.join(tmp_dir, 'test_not_there'), test_object0)
    check_eq(new_object, test_object0)
    with self.assertRaises(ValueError):
      checkpoints.restore_checkpoint(
          tmp_dir, test_object0, step=5, prefix='test_')

  @parameterized.parameters({'use_orbax': True}, {'use_orbax': False})
  def test_overwrite_checkpoints(self, use_orbax):
    config.update('flax_use_orbax_checkpointing', use_orbax)
    overwrite_error = ValueError if use_orbax else errors.InvalidCheckpointError
    tmp_dir = self.create_tempdir().full_path
    test_object0 = {'a': np.array([0, 0, 0], np.int32)}
    test_object = {'a': np.array([1, 2, 3], np.int32)}

    checkpoints.save_checkpoint(tmp_dir, test_object0, 0, keep=1)
    with self.assertRaises(overwrite_error):
      checkpoints.save_checkpoint(tmp_dir, test_object, 0, keep=1)
    checkpoints.save_checkpoint(tmp_dir, test_object, 0, keep=1, overwrite=True)
    new_object = checkpoints.restore_checkpoint(tmp_dir, test_object0)
    check_eq(new_object, test_object)

    os.chdir(os.path.dirname(tmp_dir))
    rel_tmp_dir = './' + os.path.basename(tmp_dir)
    checkpoints.save_checkpoint(rel_tmp_dir, test_object, 3, keep=1)
    new_object = checkpoints.restore_checkpoint(rel_tmp_dir, test_object0)
    check_eq(new_object, test_object)
    non_norm_dir_path = tmp_dir + '//'
    checkpoints.save_checkpoint(non_norm_dir_path, test_object, 4, keep=1)
    new_object = checkpoints.restore_checkpoint(non_norm_dir_path, test_object0)
    check_eq(new_object, test_object)

  @parameterized.parameters({'use_orbax': True, 'keep_every_n_steps': None},
                            {'use_orbax': False, 'keep_every_n_steps': 7})
  def test_keep(self, use_orbax, keep_every_n_steps):
    config.update('flax_use_orbax_checkpointing', use_orbax)
    tmp_dir = self.create_tempdir().full_path
    test_object = {'a': np.array([1, 2, 3], np.int32)}
    steps_start = 17
    steps_end = 37
    keep = 3
    increment = 5

    for step in range(steps_start, steps_end, increment):
      checkpoints.save_checkpoint(tmp_dir,
                                  test_object,
                                  step=step,
                                  keep=keep,
                                  keep_every_n_steps=keep_every_n_steps)

    last_checkpoint = -float('inf')
    for step in range(steps_start, steps_end, increment):
      if ((steps_end - step) / increment <= keep) or (keep_every_n_steps and (
          step - last_checkpoint) >= keep_every_n_steps):
        restored = checkpoints.restore_checkpoint(
            tmp_dir, target=None, step=step)
        check_eq(restored, test_object)
        last_checkpoint = step
      else:
        with self.assertRaises(ValueError):
          checkpoints.restore_checkpoint(tmp_dir, target=None, step=step)

  @parameterized.parameters({'use_orbax': True}, {'use_orbax': False})
  def test_save_restore_checkpoints_w_float_steps(self, use_orbax):
    config.update('flax_use_orbax_checkpointing', use_orbax)
    tmp_dir = self.create_tempdir().full_path
    test_object0 = {'a': np.array([0, 0, 0], np.int32),
                    'b': np.array([0, 0, 0], np.int32)}
    test_object1 = {'a': np.array([1, 2, 3], np.int32),
                    'b': np.array([1, 1, 1], np.int32)}
    test_object2 = {'a': np.array([4, 5, 6], np.int32),
                    'b': np.array([2, 2, 2], np.int32)}
    # Create leftover temporary checkpoint, which should be ignored.
    io.GFile(os.path.join(tmp_dir, 'test_tmp'), 'w')
    checkpoints.save_checkpoint(
        tmp_dir, test_object1, 0.0, prefix='test_', keep=1)
    self.assertIn('test_0.0', os.listdir(tmp_dir))
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, prefix='test_')
    check_eq(new_object, test_object1)
    checkpoints.save_checkpoint(
        tmp_dir, test_object1, 2.0, prefix='test_', keep=1)
    checkpoints.save_checkpoint(
        tmp_dir, test_object2, 3.0, prefix='test_', keep=2)
    self.assertIn('test_3.0', os.listdir(tmp_dir))
    self.assertIn('test_2.0', os.listdir(tmp_dir))
    check_eq(new_object, test_object1)

  @parameterized.parameters({'use_orbax': True}, {'use_orbax': False})
  def test_save_restore_checkpoints_target_none(self, use_orbax):
    config.update('flax_use_orbax_checkpointing', use_orbax)
    tmp_dir = self.create_tempdir().full_path
    test_object0 = {'a': np.array([0, 0, 0], np.int32),
                    'b': np.array([0, 0, 0], np.int32)}
    # Target pytree is a dictionary, so it's equal to a restored state_dict.
    checkpoints.save_checkpoint(tmp_dir, test_object0, 0)
    new_object = checkpoints.restore_checkpoint(tmp_dir, target=None)
    check_eq(new_object, test_object0)
    # Target pytree it's a tuple, check the expected state_dict is recovered.
    test_object1 = (np.array([0, 0, 0], np.int32),
                    np.array([1, 1, 1], np.int32))
    checkpoints.save_checkpoint(tmp_dir, test_object1, 1)
    new_object = checkpoints.restore_checkpoint(tmp_dir, target=None)
    expected_new_object = {str(k): v for k, v in enumerate(test_object1)}
    check_eq(new_object, expected_new_object)

  def test_save_restore_checkpoints_target_singular(self):
    tmp_dir = self.create_tempdir().full_path
    test_object0 = np.array([0, 0, 0], np.int32)
    test_object1 = np.array([1, 1, 1], np.int32)
    checkpoints.save_checkpoint(tmp_dir, test_object1, 0)
    new_object = checkpoints.restore_checkpoint(tmp_dir, target=None)
    check_eq(new_object, test_object1)
    checkpoints.save_checkpoint(tmp_dir, test_object0, 1)
    new_object = checkpoints.restore_checkpoint(tmp_dir, target=test_object1)
    check_eq(new_object, test_object0)

  @parameterized.parameters({'use_orbax': True}, {'use_orbax': False})
  def test_save_restore_checkpoints_target_empty(self, use_orbax):
    config.update('flax_use_orbax_checkpointing', use_orbax)
    tmp_dir = self.create_tempdir().full_path
    test_object0 = {}
    test_object1 = []
    # Orbax returns ValueError if the target is empty, but legacy Flax doesn't.
    if use_orbax:
      with self.assertRaises(ValueError):
        checkpoints.save_checkpoint(tmp_dir, test_object1, 0)
    else:
      checkpoints.save_checkpoint(tmp_dir, test_object1, 0)
      new_object = checkpoints.restore_checkpoint(tmp_dir, target=None)
      check_eq(new_object, test_object0)
      checkpoints.save_checkpoint(tmp_dir, test_object0, 1)
      new_object = checkpoints.restore_checkpoint(tmp_dir, target=test_object1)
      check_eq(new_object, test_object1)

  def test_async_save_checkpoints(self):
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    test_object0 = {'a': np.array([0, 0, 0], np.int32),
                    'b': np.array([0, 0, 0], np.int32)}
    test_object1 = {'a': np.random.normal(size=(1000, 1000)),
                    'b': np.random.normal(size=(1000, 1000))}
    test_object2 = {'a': np.random.normal(size=(1000, 1000)),
                    'b': np.random.normal(size=(1000, 1000))}
    test_object3 = {'a': np.random.normal(size=(1000, 1000)),
                    'b': np.random.normal(size=(1000, 1000))}
    am = checkpoints.AsyncManager()
    checkpoints.save_checkpoint(
        tmp_dir, test_object1, 0, prefix='test_', keep=1, async_manager=am)
    # Hard-wait the write to be done, then check its content.
    am.save_future.result()
    self.assertIn('test_0', os.listdir(tmp_dir))
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object1, prefix='test_')
    check_eq(new_object, test_object1)
    # Check two consecutive saves happen in the right order.
    checkpoints.save_checkpoint(
        tmp_dir, test_object2, 1, prefix='test_', keep=1, async_manager=am)
    checkpoints.save_checkpoint(
        tmp_dir, test_object3, 2, prefix='test_', keep=1, async_manager=am)
    am.save_future.result()
    self.assertIn('test_2', os.listdir(tmp_dir))
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object1, prefix='test_')
    check_eq(new_object, test_object3)

  def test_last_checkpoint(self):
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    with io.GFile(os.path.join(tmp_dir, 'test_tmp'), 'w') as f:
      f.write('test_tmp')
    io.makedirs(os.path.join(tmp_dir, 'test_tmp_gda'))
    self.assertEqual(checkpoints.latest_checkpoint(tmp_dir, 'test_'), None)

    with io.GFile(os.path.join(tmp_dir, 'test_0'), 'w') as f:
      f.write('test_0')
    io.makedirs(os.path.join(tmp_dir, 'test_0_gda'))
    self.assertEqual(
        checkpoints.latest_checkpoint(tmp_dir, 'test_'),
        os.path.join(tmp_dir, 'test_0'),
    )

    with io.GFile(os.path.join(tmp_dir, 'test_10'), 'w') as f:
      f.write('test_10')
    self.assertEqual(
        checkpoints.latest_checkpoint(tmp_dir, 'test_'),
        os.path.join(tmp_dir, 'test_10'),
    )
    self.assertEqual(checkpoints.latest_checkpoint(tmp_dir, 'ckpt_'), None)

    path = f'orbaxtest_{orbax.utils.TMP_DIR_SUFFIX}_10'
    with io.GFile(os.path.join(tmp_dir, path), 'w') as f:
      f.write('orbaxtest_10')
    self.assertIsNone(checkpoints.latest_checkpoint(tmp_dir, 'orbaxtest_'))

  @parameterized.parameters(
      {'step_type': int, 'steps': [1, 5, 112]},
      {'step_type': float, 'steps': [1.0, 4.5, 5.6]},
  )
  def test_available_steps(self, step_type, steps):
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    with io.GFile(os.path.join(tmp_dir, 'test_tmp'), 'w') as f:
      f.write('test_tmp')
    io.makedirs(os.path.join(tmp_dir, 'test_tmp_gda'))

    for step in steps:
      with io.GFile(os.path.join(tmp_dir, 'test_' + str(step)), 'w') as f:
        f.write('test_' + str(step))
      io.makedirs(os.path.join(tmp_dir, 'test_' + str(step) + '_gda'))

    self.assertEqual(
        checkpoints.available_steps(tmp_dir, 'test_', step_type=step_type),
        steps,
    )

  @parameterized.parameters({'use_orbax': True}, {'use_orbax': False})
  def test_complex_pytree(self, use_orbax):
    config.update('flax_use_orbax_checkpointing', use_orbax)
    tmp_dir = self.create_tempdir().full_path
    to_save = [
        CustomDC(foo=12, bar=core.freeze({'x': jnp.array((1, 4))})),
        np.array((2, 3)),
    ]
    target = [
        CustomDC(foo=0, bar=core.freeze({'x': jnp.array((0, 0))})),
        np.array((0, 0)),
    ]
    checkpoints.save_checkpoint(tmp_dir, to_save, 0)
    restored = checkpoints.restore_checkpoint(tmp_dir, target=target)
    check_eq(restored, to_save)

  # restore_checkpoint can automatically restore either orbax or legacy files.
  def test_auto_restore(self):
    tmp_dir = self.create_tempdir().full_path
    to_save = [CustomDC(foo=12, bar={'x': jnp.array((1, 4))}), np.array((2, 3))]
    target = [CustomDC(foo=0, bar={'x': jnp.array((0, 0))}), np.array((0, 0))]
    # Store an orbax ckpt
    config.update('flax_use_orbax_checkpointing', True)
    checkpoints.save_checkpoint(tmp_dir, to_save, 0, prefix='test_')
    # And a legacy ckpt
    config.update('flax_use_orbax_checkpointing', False)
    checkpoints.save_checkpoint(tmp_dir, to_save, 1, prefix='test_', keep=2)

    # Both gets restored with same API.
    restored = checkpoints.restore_checkpoint(
        os.path.join(tmp_dir, 'test_0'), target=target)
    check_eq(restored, to_save)
    restored = checkpoints.restore_checkpoint(
        os.path.join(tmp_dir, 'test_1'), target=target)
    check_eq(restored, to_save)

  def test_convert_pre_linen(self):
    params = checkpoints.convert_pre_linen({
        'mod_0': {
            'submod1_0': {},
            'submod2_1': {},
            'submod1_2': {},
        },
        'mod2_2': {
            'submod2_2_0': {}
        },
        'mod2_11': {
            'submod2_11_0': {}
        },
        'mod2_1': {
            'submod2_1_0': {}
        },
    })
    self.assertDictEqual(
        core.unfreeze(params), {
            'mod_0': {
                'submod1_0': {},
                'submod1_1': {},
                'submod2_0': {},
            },
            'mod2_0': {
                'submod2_1_0': {}
            },
            'mod2_1': {
                'submod2_2_0': {}
            },
            'mod2_2': {
                'submod2_11_0': {}
            },
        })


if __name__ == '__main__':
  absltest.main()
