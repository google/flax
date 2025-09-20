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

from unittest import mock

from absl.testing import absltest

from flax.configurations import bool_flag, config


class MyTestCase(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.object(config, '_values', {}))
    self._flag = bool_flag('test', default=False, help='Just a test flag.')

  def test_duplicate_flag(self):
    with self.assertRaisesRegex(RuntimeError, 'already defined'):
      bool_flag(self._flag.name, default=False, help='Another test flag.')

  def test_default(self):
    self.assertFalse(self._flag.value)
    self.assertFalse(config.test)

  def test_typed_update(self):
    config.update(self._flag, True)
    self.assertTrue(self._flag.value)
    self.assertTrue(config.test)

  def test_untyped_update(self):
    config.update(self._flag.name, True)
    self.assertTrue(self._flag.value)
    self.assertTrue(config.test)

  def test_update_unknown_flag(self):
    with self.assertRaisesRegex(LookupError, 'Unrecognized config option'):
      config.update('unknown', True)


if __name__ == '__main__':
  absltest.main()
