# Copyright 2021 The Flax Authors.
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
from typing import Any

from absl.testing import absltest
import flax
from flax import core
from flax import errors
from flax.training import checkpoints
import jax
from jax import numpy as jnp
from jax import test_util as jtu
import numpy as np
from tensorflow.io import gfile

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


def shuffle(l):
  """Functional shuffle."""
  l = copy.copy(l)
  np.random.shuffle(l)
  return l


class InnerPreLinen(flax.nn.Module):
  """Inner class based on pre-Linen flax.nn."""

  def apply(self, x):
    x = flax.nn.Conv(x, 10, (2, 2))
    x = flax.nn.normalization.BatchNorm(x, use_running_average=True)
    return x


class ModelPreLinen(flax.nn.Module):
  """Simple model based on pre-Linen flax.nn."""

  def apply(self, inputs):
    x = flax.nn.Conv(inputs, 10, (2, 2))
    x = InnerPreLinen(x, name='Inner_1')
    x = x.reshape([x.shape[0], -1])
    x = flax.nn.normalization.BatchNorm(x, use_running_average=True)
    x = flax.nn.Dense(x, 10)
    x = flax.nn.log_softmax(x)
    return x


class Inner(flax.linen.Module):
  """Inner class based on flax.linen."""

  @flax.linen.compact
  def __call__(self, x):
    x = flax.linen.Conv(10, (2, 2))(x)
    x = flax.linen.normalization.BatchNorm(True)(x)
    return x


class Model(flax.linen.Module):
  """Simple model based on flax.linen."""

  @flax.linen.compact
  def __call__(self, inputs):
    x = flax.linen.Conv(10, (2, 2))(inputs)
    x = Inner()(x)
    x = x.reshape([x.shape[0], -1])
    x = flax.linen.normalization.BatchNorm(True)(x)
    x = flax.linen.Dense(10)(x)
    x = flax.linen.log_softmax(x)
    return x


@flax.struct.dataclass
class TrainState:
  """Simple container that captures training state."""
  optimizer: flax.optim.Optimizer
  model_state: Any


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
    # Restore a specific path.
    new_object = checkpoints.restore_checkpoint(
        os.path.join(tmp_dir, 'test_3'), test_object0)
    jtu.check_eq(new_object, test_object2)
    # If a specific path is specified, but it does not exist, the same behavior
    # as when a directory is empty should apply: the target is returned
    # unchanged.
    new_object = checkpoints.restore_checkpoint(
        os.path.join(tmp_dir, 'test_not_there'), test_object0)
    jtu.check_eq(new_object, test_object0)
    with self.assertRaises(ValueError):
      checkpoints.restore_checkpoint(
          tmp_dir, test_object0, step=5, prefix='test_')

  def test_overwrite_checkpoints(self):
    tmp_dir = self.create_tempdir().full_path
    test_object0 = {'a': np.array([0, 0, 0], np.int32)}
    test_object = {'a': np.array([1, 2, 3], np.int32)}

    checkpoints.save_checkpoint(
        tmp_dir, test_object0, 0, keep=1)
    with self.assertRaises(errors.InvalidCheckpointError):
      checkpoints.save_checkpoint(
          tmp_dir, test_object, 0, keep=1)
    checkpoints.save_checkpoint(
          tmp_dir, test_object, 0, keep=1, overwrite=True)
    new_object = checkpoints.restore_checkpoint(tmp_dir, test_object0)
    jtu.check_eq(new_object, test_object)
    checkpoints.save_checkpoint(
          tmp_dir, test_object0, 2, keep=1, overwrite=True)
    new_object = checkpoints.restore_checkpoint(tmp_dir, test_object)
    jtu.check_eq(new_object, test_object0)
    with self.assertRaises(errors.InvalidCheckpointError):
      checkpoints.save_checkpoint(
            tmp_dir, test_object, 1, keep=1)
    checkpoints.save_checkpoint(
          tmp_dir, test_object, 1, keep=1, overwrite=True)
    new_object = checkpoints.restore_checkpoint(tmp_dir, test_object0)
    jtu.check_eq(new_object, test_object)

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
    with self.assertRaises(errors.InvalidCheckpointError):
      checkpoints.save_checkpoint(
          tmp_dir, test_object2, 1.0, prefix='test_', keep=1)
    checkpoints.save_checkpoint(
        tmp_dir, test_object2, 3.0, prefix='test_', keep=2)
    self.assertIn('test_3.0', os.listdir(tmp_dir))
    self.assertIn('test_2.0', os.listdir(tmp_dir))
    jtu.check_eq(new_object, test_object1)

  def test_save_restore_checkpoints_target_none(self):
    tmp_dir = self.create_tempdir().full_path
    test_object0 = {'a': np.array([0, 0, 0], np.int32),
                    'b': np.array([0, 0, 0], np.int32)}
    # Target pytree is a dictionary, so it's equal to a restored state_dict.
    checkpoints.save_checkpoint(tmp_dir, test_object0, 0)
    new_object = checkpoints.restore_checkpoint(tmp_dir, target=None)
    jtu.check_eq(new_object, test_object0)
    # Target pytree it's a tuple, check the expected state_dict is recovered.
    test_object1 = (np.array([0, 0, 0], np.int32),
                    np.array([1, 1, 1], np.int32))
    checkpoints.save_checkpoint(tmp_dir, test_object1, 1)
    new_object = checkpoints.restore_checkpoint(tmp_dir, target=None)
    expected_new_object = {str(k): v for k, v in enumerate(test_object1)}
    jtu.check_eq(new_object, expected_new_object)

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

  def test_convert_checkpoint(self):
    inputs = jnp.ones([2, 5, 5, 1])
    rng = jax.random.PRNGKey(0)
    # pre-Linen.
    with flax.nn.stateful() as model_state:
      y, params = ModelPreLinen.init(rng, inputs)
    pre_linen_optimizer = flax.optim.GradientDescent(0.1).create(params)
    train_state = TrainState(
        optimizer=pre_linen_optimizer, model_state=model_state)
    state_dict = flax.serialization.to_state_dict(train_state)
    # Linen.
    model = Model()
    variables = model.init(rng, inputs)
    optimizer = flax.optim.GradientDescent(0.1).create(variables['params'])
    optimizer = optimizer.restore_state(
        flax.core.unfreeze(
            checkpoints.convert_pre_linen(state_dict['optimizer'])))
    optimizer = optimizer.apply_gradient(variables['params'])
    batch_stats = checkpoints.convert_pre_linen(
        flax.traverse_util.unflatten_dict({
            tuple(k.split('/')[1:]): v
            for k, v in model_state.as_dict().items()
        }))
    y, updated_state = model.apply(
        dict(params=optimizer.target, batch_stats=batch_stats),
        inputs,
        mutable=['batch_stats'])
    del y, updated_state  # not used.


if __name__ == '__main__':
  absltest.main()
