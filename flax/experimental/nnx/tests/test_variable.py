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

import typing as tp

import jax

from flax.experimental import nnx

A = tp.TypeVar('A')


class TestVariable:
  def test_value(self):
    r1 = nnx.Variable(1)
    assert r1.raw_value == 1

    r2 = jax.tree_map(lambda x: x + 1, r1)

    assert r1.raw_value == 1
    assert r2.raw_value == 2
    assert r1 is not r2
