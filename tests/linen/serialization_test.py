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

"""Tests for flax.serialization."""

from absl.testing import absltest, parameterized
from flax import serialization
import jax


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class SerializationTest(absltest.TestCase):

  def test_from_state_dict_checks(self):
    class UnkownType:
      pass
    state_dict = UnkownType()
    with self.assertRaisesRegex(ValueError, 'No serialization handler for target of type'):
      serialization.from_state_dict(UnkownType, state_dict)
    
    output = serialization.from_state_dict(None, state_dict)
    self.assertEqual(output, state_dict)