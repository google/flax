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

"""Flax test."""

from absl import app

import flax


@flax.struct.dataclass
class TrainState():
  history: int
  rng: int
  step: int
  metrics: int


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # This gives an error:
  # Invalid keyword arguments (history, metrics, rng, step) to function
  # TrainState.__init__ [wrong-keyword-args]
  #        Expected: (self)
  # Actually passed: (self, history, metrics, rng, step)
  #
  # Is this a bug?

  t = TrainState(history=0, rng=1, step=2, metrics=3)


if __name__ == '__main__':
  app.run(main)
