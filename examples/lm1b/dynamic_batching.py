# Lint as: python3

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

"""A general purpose dynamic batching utility.

In particular, this can be used as part of a language model input pipeline.
"""

import bisect


def dynamic_batches(generator, bucket_lengths, batch_capacity):
  """Batch lists of tokens ("sentences") into batches of dynamic sizes.

  1. Collect sentences into buckets, where each bucket has examples up to
     a certain length.

  2. Once the total number of tokens in a bucket would surpass a pre-defined
     total batch capacity, the bucket is turned into a batch and yielded.

  3. The elements in each batch are padded to the maximum length of examples
     in that batch. Since similar lengthed sentence are grouped together, the
     amount of padding can be kept small.

  4. Also emit a mask (a list of 1's followed by 0's) corresponding to the
     part of the padded sentence that was the original sentence.

  Each batch ends up having a different number of examples and a different
  sentence length. For example, with batch capacity 4096, one batch may have 64
  example with 64 tokens each, and another batch may have 146 examples
  of 28 tokens each.

  This is particularly important for JAX jitted functions, which compile new
  programs for each input shape to the function. For each of the finitely
  many bucket_lengths, there is a single possible batch size. So the total
  number of possible distinct batch shapes emitted from this generator
  is the same as len(bucket_lengths).

  Args:
    generator: A generator yielding sentences.
    bucket_lengths: A list of the maximum sentence length in each bucket.
    batch_capacity: The total maximum number of tokens in each batch.

  Yields:
    A dict e.g. {
        'bucket_length': 6
        'padded': [[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 0, 0]],
        'mask': [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]]
    }

  TODO(avitalo@): Figure out how this works in multi-host settings, like
  TPU pods. Do we need to dispatch a batch of the same size to each host?
  """
  buffers = [{'padded': [], 'mask': []}
             for _ in range(len(bucket_lengths))]

  assert all(
      (bucket_length <= batch_capacity for bucket_length in bucket_lengths))

  for text in generator:
    length = len(text)
    bucket_idx = bisect.bisect_left(bucket_lengths, length)
    buffer = buffers[bucket_idx]
    bucket_length = bucket_lengths[bucket_idx]

    # TODO(avitalo@): Use numpy utilities such as np.pad(). In order to do
    # this, we need to find or build a parallel to assertSameStructure
    # in the test that is numpy-aware.
    buffer['padded'].append(_zero_pad(text, bucket_length))
    buffer['mask'].append(_zero_pad([1] * length, bucket_length))

    # If the buffer can't fit another item of the same length, flush it.
    if len(buffer['padded']) == batch_capacity // bucket_length:
      yield {
          'bucket_length': bucket_length,
          'padded': buffer['padded'],
          'mask': buffer['mask'],
      }

      buffer['text'] = []
      buffer['padded'] = []
      buffer['mask'] = []


def _zero_pad(v, capacity):
  assert capacity >= len(v), '`v` is shorter than `capacity`'
  return v + ([0] * (capacity - len(v)))

