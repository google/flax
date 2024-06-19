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

"""Tests for flax.errors."""

from absl.testing import absltest
from flax.errors import FlaxError, ScopeVariableNotFoundError
import pickle

class ErrorrsTest(absltest.TestCase):
  def test_exception_can_be_pickled(self):
    # tests the new __reduce__ method fixes bug reported in issue #4000
    ex = ScopeVariableNotFoundError('varname', 'collection', 'scope')
    pickled_ex = pickle.dumps(ex)
    unpicked_ex = pickle.loads(pickled_ex)
    self.assertIsInstance(unpicked_ex, FlaxError)
    self.assertIn('varname', str(unpicked_ex))
    self.assertIn('#flax.errors.ScopeVariableNotFoundError', str(unpicked_ex))
    self.assertNotIn('#flax.errors.FlaxError', str(unpicked_ex))


if __name__ == '__main__':
  absltest.main()
