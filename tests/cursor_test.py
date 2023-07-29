# Copyright 2023 The Flax Authors.
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

"""Tests for flax.struct."""


from absl.testing import absltest
import jax
import optax

from flax.core import freeze
from flax.cursor import cursor
from flax.training import train_state

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class CursorTest(absltest.TestCase):

  def test_basic(self):
    # EXMAPLE
    t = freeze({'a': 1, 'b': (2, 3), 'c': [4, 5]})

    # set API
    print(cursor(t)['b'][0].set(10))
    print()

    # build API
    c = cursor(t)
    c['b'][0] = 10
    c['a'] = (100, 200)
    t2 = c.build()

    print(t2)
    print()

    state = train_state.TrainState.create(
        apply_fn=lambda x: x,
        params=t,
        tx=optax.adam(1e-3),
    )
    print(cursor(state).params['b'][0].set(10))
