# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for flax.examples.lm1b.input_pipeline."""

from absl.testing import absltest
from flax.examples.lm1b import dynamic_batching


class InputPipelineTest(absltest.TestCase):

  def test_batch_size_fits_buckets(self):
    """Test that we assert if batch_capacity < some bucket length."""
    with self.assertRaises(AssertionError):
      for _ in dynamic_batching.dynamic_batches(
          generator=[],
          bucket_lengths=[2, 4, 8],
          batch_capacity=6):
        pass

  def test_bucketing(self):
    sentences = [
        # should fit in bucket length 4
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10],  # this sequence shouldn't fit in any bucket

        # should fit in bucket length 2
        [11],
        [12, 13],

        # should fit in bucket length 8
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30],

        # should fit in bucket length 2 (cont.)
        [14, 15],
        [16, 17],
        [18, 19],  # this sequence shouldn't fit in any bucket
    ]

    batches_generator = dynamic_batching.dynamic_batches(
        sentences, bucket_lengths=[2, 4, 8], batch_capacity=8)
    batches = list(batches_generator)

    self.assertSameStructure(batches, [{
        'bucket_length': 4,
        'padded': [[1, 2, 3, 0], [4, 5, 6, 7]],
        'mask': [[1, 1, 1, 0], [1, 1, 1, 1]],
    }, {
        'bucket_length': 8,
        'padded': [[20, 21, 22, 23, 24, 0, 0, 0]],
        'mask': [[1, 1, 1, 1, 1, 0, 0, 0]],
    }, {
        'bucket_length': 8,
        'padded': [[25, 26, 27, 28, 29, 30, 0, 0]],
        'mask': [[1, 1, 1, 1, 1, 1, 0, 0]],
    }, {
        'bucket_length': 2,
        'padded': [[11, 0], [12, 13], [14, 15], [16, 17]],
        'mask': [[1, 0], [1, 1], [1, 1], [1, 1]],
    }])


if __name__ == '__main__':
  absltest.main()
