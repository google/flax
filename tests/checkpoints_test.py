# Copyright 2020 The Flax Authors.
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

from absl.testing import absltest
from flax.training import checkpoints
import jax
from jax import test_util as jtu
import numpy as np
from tensorflow.compat.v2.io import gfile

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


def shuffle(l):
  """Functional shuffle."""
  l = copy.copy(l)
  np.random.shuffle(l)
  return l


class CheckpointsTest(absltest.TestCase):

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

  def test_save_restore_checkpoints(self):
    tmp_dir = self.create_tempdir().full_path
    test_object0 = {'a': np.array([0, 0, 0], np.int32),
                    'b': np.array([0, 0, 0], np.int32)}
    test_object1 = {'a': np.array([1, 2, 3], np.int32),
                    'b': np.array([1, 1, 1], np.int32)}
    test_object2 = {'a': np.array([4, 5, 6], np.int32),
                    'b': np.array([2, 2, 2], np.int32)}
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, prefix='test_')
    jtu.check_eq(new_object, test_object0)
    # Create leftover temporary checkpoint, which should be ignored.
    gfile.GFile(os.path.join(tmp_dir, 'test_tmp'), 'w')
    checkpoints.save_checkpoint(
        tmp_dir, test_object1, 0, prefix='test_', keep=1)
    self.assertIn('test_0', os.listdir(tmp_dir))
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, prefix='test_')
    jtu.check_eq(new_object, test_object1)
    checkpoints.save_checkpoint(
        tmp_dir, test_object1, 1, prefix='test_', keep=1)
    checkpoints.save_checkpoint(
        tmp_dir, test_object2, 2, prefix='test_', keep=1)
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, prefix='test_')
    jtu.check_eq(new_object, test_object2)
    checkpoints.save_checkpoint(
        tmp_dir, test_object2, 3, prefix='test_', keep=2)
    checkpoints.save_checkpoint(
        tmp_dir, test_object1, 4, prefix='test_', keep=2)
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, prefix='test_')
    jtu.check_eq(new_object, test_object1)
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, step=3, prefix='test_')
    jtu.check_eq(new_object, test_object2)
    with self.assertRaises(ValueError):
      checkpoints.restore_checkpoint(
          tmp_dir, test_object0, step=5, prefix='test_')

  def test_save_restore_checkpoints_w_float_steps(self):
    tmp_dir = self.create_tempdir().full_path
    test_object0 = {'a': np.array([0, 0, 0], np.int32),
                    'b': np.array([0, 0, 0], np.int32)}
    test_object1 = {'a': np.array([1, 2, 3], np.int32),
                    'b': np.array([1, 1, 1], np.int32)}
    test_object2 = {'a': np.array([4, 5, 6], np.int32),
                    'b': np.array([2, 2, 2], np.int32)}
    # Create leftover temporary checkpoint, which should be ignored.
    gfile.GFile(os.path.join(tmp_dir, 'test_tmp'), 'w')
    checkpoints.save_checkpoint(
        tmp_dir, test_object1, 0.0, prefix='test_', keep=1)
    self.assertIn('test_0.0', os.listdir(tmp_dir))
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, prefix='test_')
    jtu.check_eq(new_object, test_object1)
    checkpoints.save_checkpoint(
        tmp_dir, test_object1, 2.0, prefix='test_', keep=1)
    checkpoints.save_checkpoint(
        tmp_dir, test_object2, 1.0, prefix='test_', keep=1)
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, prefix='test_')
    jtu.check_eq(new_object, test_object1)
    checkpoints.save_checkpoint(
        tmp_dir, test_object2, 3.0, prefix='test_', keep=2)
    checkpoints.save_checkpoint(
        tmp_dir, test_object1, -1.0, prefix='test_', keep=2)
    new_object = checkpoints.restore_checkpoint(
        tmp_dir, test_object0, prefix='test_')
    self.assertIn('test_3.0', os.listdir(tmp_dir))
    self.assertIn('test_2.0', os.listdir(tmp_dir))
    jtu.check_eq(new_object, test_object2)


if __name__ == '__main__':
  absltest.main()
