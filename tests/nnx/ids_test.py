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

import copy

from absl.testing import absltest
from flax.nnx import ids


class TestIds(absltest.TestCase):
  def test_hashable(self):
    id1 = ids.uuid()
    id2 = ids.uuid()
    assert id1 == id1
    assert id1 != id2
    assert hash(id1) != hash(id2)
    id1c = copy.copy(id1)
    id1dc = copy.deepcopy(id1)
    assert hash(id1) != hash(id1c)
    assert hash(id1) != hash(id1dc)


if __name__ == '__main__':
  absltest.main()
