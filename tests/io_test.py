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

"""Tests for flax.io."""

import tempfile
import os

from absl.testing import absltest
from absl.testing import parameterized
from flax import io
from flax import errors
import tensorflow as tf
import jax

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class IOTest(parameterized.TestCase):

  @parameterized.parameters(
    {'backend_mode': io.BackendMode.DEFAULT},
    {'backend_mode': io.BackendMode.TF}
  )
  def test_override(self, backend_mode):
    with io.override_mode(backend_mode):
      self.assertEqual(io.io_mode, backend_mode)

  @parameterized.parameters(
    {'write_mode': io.BackendMode.DEFAULT, 'read_mode': io.BackendMode.TF},
    {'write_mode': io.BackendMode.TF, 'read_mode': io.BackendMode.DEFAULT}
  )
  def test_GFile(self, write_mode, read_mode):
    test_string = b'testing write and read'
    with tempfile.TemporaryDirectory() as temp_dir_path:
      test_path = os.path.join(temp_dir_path, 'test')

      with io.override_mode(write_mode):
        with io.GFile(test_path, 'wb') as file:
          file.write(test_string)

      with io.override_mode(read_mode):
        with io.GFile(test_path, 'rb') as file:
          self.assertEqual(file.read(), test_string)

  def test_listdir(self):
    with tempfile.TemporaryDirectory() as temp_dir_path:
      os.mkdir(os.path.join(temp_dir_path, 'a'))
      os.mkdir(os.path.join(temp_dir_path, 'as'))
      os.mkdir(os.path.join(temp_dir_path, 'af'))
      os.mkdir(os.path.join(temp_dir_path, 'test'))
      os.mkdir(os.path.join(temp_dir_path, 'at'))

      with io.override_mode(io.BackendMode.DEFAULT):
        default_dir_set = set(io.listdir(temp_dir_path))

      with io.override_mode(io.BackendMode.TF):
        tf_dir_set = set(io.listdir(temp_dir_path))

      self.assertEqual(default_dir_set, tf_dir_set)

  @parameterized.parameters(
    {'create_temp_fn': tempfile.TemporaryDirectory},
    {'create_temp_fn': tempfile.NamedTemporaryFile}
  )
  def test_isdir(self, create_temp_fn):
    with create_temp_fn() as temp:
      path = temp.name if hasattr(temp, 'name') else temp

      with io.override_mode(io.BackendMode.DEFAULT):
        default_isdir = io.isdir(path)

      with io.override_mode(io.BackendMode.TF):
        tf_isdir = io.isdir(path)

      self.assertEqual(default_isdir, tf_isdir)

  def test_copy(self):
    test_string = b'testing copy'
    with tempfile.TemporaryDirectory() as temp_dir_path:
      test_path = os.path.join(temp_dir_path, 'test')
      copy1_path = os.path.join(temp_dir_path, 'copy1')
      copy2_path = os.path.join(temp_dir_path, 'copy2')

      with io.GFile(test_path, 'wb') as file:
        file.write(test_string)

      with io.override_mode(io.BackendMode.DEFAULT):
        io.copy(test_path, copy1_path)

      with io.override_mode(io.BackendMode.TF):
        io.copy(copy1_path, copy2_path)

      with io.GFile(copy2_path, 'rb') as file:
        self.assertEqual(file.read(), test_string)

  @parameterized.parameters(
    {'backend_mode': io.BackendMode.DEFAULT, 'error_type': errors.AlreadyExistsError},
    {'backend_mode': io.BackendMode.TF, 'error_type': tf.errors.AlreadyExistsError},
  )
  def test_copy_raises_error(self, backend_mode, error_type):
    with tempfile.NamedTemporaryFile() as temp_file:
      with io.override_mode(backend_mode):
        with self.assertRaises(error_type):
          io.copy(temp_file.name, temp_file.name)

  def test_rename(self):
    with tempfile.TemporaryDirectory() as temp_dir_path:
      test_path = os.path.join(temp_dir_path, 'test')
      rename1_path = os.path.join(temp_dir_path, 'rename1')
      rename2_path = os.path.join(temp_dir_path, 'rename2')

      with io.GFile(test_path, 'wb') as file:
        file.write(b'placeholder text')

      with io.override_mode(io.BackendMode.DEFAULT):
        io.rename(test_path, rename1_path)

      with io.override_mode(io.BackendMode.TF):
        io.rename(rename1_path, rename2_path)

      with io.GFile(rename2_path, 'rb') as file:
        self.assertTrue(os.path.exists(rename2_path))

  @parameterized.parameters(
    {'backend_mode': io.BackendMode.DEFAULT, 'error_type': errors.AlreadyExistsError},
    {'backend_mode': io.BackendMode.TF, 'error_type': tf.errors.AlreadyExistsError},
  )
  def test_rename_raises_error(self, backend_mode, error_type):
    with tempfile.NamedTemporaryFile() as temp_file:
      with io.override_mode(backend_mode):
        with self.assertRaises(error_type):
          io.rename(temp_file.name, temp_file.name)

  def test_exists(self):
    with tempfile.NamedTemporaryFile() as temp_file:

      with io.override_mode(io.BackendMode.DEFAULT):
        default_exists = io.exists(temp_file.name)

      with io.override_mode(io.BackendMode.TF):
        tf_exists = io.exists(temp_file.name)

      self.assertEqual(default_exists, tf_exists)

  @parameterized.parameters(
    {'backend_mode': io.BackendMode.DEFAULT},
    {'backend_mode': io.BackendMode.TF}
  )
  def test_makedirs(self, backend_mode):
    with tempfile.TemporaryDirectory() as temp_dir_path:
      test_dir_path = os.path.join(temp_dir_path, 'test_dir')

      with io.override_mode(backend_mode):
        io.makedirs(test_dir_path)
      self.assertTrue(os.path.exists(test_dir_path) and (os.path.isdir(test_dir_path)))

  def test_glob(self):
    with tempfile.TemporaryDirectory() as temp_dir_path:
      os.mkdir(os.path.join(temp_dir_path, 'a'))
      os.mkdir(os.path.join(temp_dir_path, 'as'))
      os.mkdir(os.path.join(temp_dir_path, 'af'))
      os.mkdir(os.path.join(temp_dir_path, 'test'))
      os.mkdir(os.path.join(temp_dir_path, 'at'))

    with io.override_mode(io.BackendMode.DEFAULT):
      default_glob_set = set(io.glob('a*/'))

    with io.override_mode(io.BackendMode.TF):
      tf_glob_set = set(io.glob('a*/'))

    self.assertEqual(default_glob_set, tf_glob_set)

  @parameterized.parameters(
    {'backend_mode': io.BackendMode.DEFAULT},
    {'backend_mode': io.BackendMode.TF}
  )
  def test_remove(self, backend_mode):
    with tempfile.TemporaryDirectory() as temp_dir_path:
      test_path = os.path.join(temp_dir_path, 'test')

      with io.GFile(test_path, 'wb') as file:
        file.write(b'placeholder text')

      with io.override_mode(backend_mode):
        io.remove(test_path)

      self.assertTrue(not os.path.exists(test_path))

  @parameterized.parameters(
    {'backend_mode': io.BackendMode.DEFAULT},
    {'backend_mode': io.BackendMode.TF}
  )
  def test_rmtree(self, backend_mode):
    with tempfile.TemporaryDirectory() as temp_dir_path:
      dir0_path = os.path.join(temp_dir_path, 'dir0')

      os.mkdir(dir0_path)
      os.mkdir(os.path.join(dir0_path, 'dir1'))
      os.mkdir(os.path.join(dir0_path, 'dir1', 'dir2'))
      os.mkdir(os.path.join(dir0_path, 'dir1', 'dir3'))
      os.mkdir(os.path.join(dir0_path, 'dir4'))
      os.mkdir(os.path.join(dir0_path, 'dir4', 'dir5'))
      os.mkdir(os.path.join(dir0_path, 'dir6'))

      with io.override_mode(backend_mode):
        io.rmtree(dir0_path)

      self.assertTrue(not os.path.exists(dir0_path))


if __name__ == '__main__':
  absltest.main()
