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

"""Input pipeline for seq2seq addition example."""

import random

import jax
import jax.numpy as jnp
import numpy as np


class CharacterTable:
  """Encodes/decodes between strings and integer representations."""

  def __init__(self, chars, max_len_query_digit=3):
    self._chars = sorted(set(chars))
    self._char_indices = dict(
        (ch, idx + 2) for idx, ch in enumerate(self._chars))
    self._indices_char = dict(
        (idx + 2, ch) for idx, ch in enumerate(self._chars))
    self._indices_char[self.pad_id] = '_'
    # Maximum length of a single input digit
    self._max_len_query_digit = max_len_query_digit

  @property
  def pad_id(self):
    return 0

  @property
  def eos_id(self):
    return 1

  @property
  def vocab_size(self):
    return len(self._chars) + 2

  @property
  def max_input_len(self):
    """Returns the max length of an input sequence."""
    return self._max_len_query_digit * 2 + 2  # includes EOS

  @property
  def max_output_len(self):
    """Returns the max length of an output sequence."""
    return self._max_len_query_digit + 3  # includes start token '=' and EOS.

  @property
  def encoder_input_shape(self):
    max_input_len = self._max_len_query_digit * 2 + 2  # includes EOS
    return (1, max_input_len, self.vocab_size)

  @property
  def decoder_input_shape(self):
    return (1, self.max_output_len, self.vocab_size)

  def encode(self, inputs):
    """Encodes from string to list of integers."""
    return np.array(
        [self._char_indices[char] for char in inputs] + [self.eos_id])

  def decode(self, inputs):
    """Decodes from list of integers to string."""
    chars = []
    for elem in inputs.tolist():
      if elem == self.eos_id:
        break
      chars.append(self._indices_char[elem])
    return ''.join(chars)

  def encode_onehot(self, batch_inputs, max_len=None):
    """One-hot encodes a string input."""

    if max_len is None:
      max_len = self.max_input_len

    def encode_str(s):
      tokens = self.encode(s)
      unpadded_len = len(tokens)
      if unpadded_len > max_len:
        raise ValueError(f'Sequence too long ({len(tokens)}>{max_len}): \'{s}\'')
      tokens = np.pad(tokens, [(0, max_len-len(tokens))], mode='constant')
      return jax.nn.one_hot(tokens, self.vocab_size, dtype=jnp.float32)

    return np.array([encode_str(inp) for inp in batch_inputs])

  def decode_onehot(self, batch_inputs):
    """Decodes a batch of one-hot encoding to strings."""
    decode_inputs = lambda inputs: self.decode(inputs.argmax(axis=-1))
    return np.array(list(map(decode_inputs, batch_inputs)))

  def generate_examples(self, num_examples):
    """Yields `num_examples` examples."""
    for _ in range(num_examples):
      max_digit = pow(10, self._max_len_query_digit) - 1
      # TODO(marcvanzee): Use jax.random here.
      key = tuple(sorted((random.randint(0, 99), random.randint(0, max_digit))))
      inputs = '{}+{}'.format(key[0], key[1])
      # Preprend output by the decoder's start token.
      outputs = '=' + str(key[0] + key[1])
      yield (inputs, outputs)

  def get_batch(self, batch_size):
    """Returns a batch of example of size @batch_size."""
    inputs, outputs = zip(*self.generate_examples(batch_size))
    return {
        'query': self.encode_onehot(inputs),
        'answer': self.encode_onehot(outputs),
    }


def mask_sequences(sequence_batch, lengths):
  """Sets positions beyond the length of each sequence to 0."""
  return sequence_batch * (
      lengths[:, np.newaxis] > np.arange(sequence_batch.shape[1])[np.newaxis])


def get_sequence_lengths(sequence_batch, eos_id):
  """Returns the length of each one-hot sequence, including the EOS token."""
  # sequence_batch.shape = (batch_size, seq_length, vocab_size)
  eos_row = sequence_batch[:, :, eos_id]
  eos_idx = jnp.argmax(eos_row, axis=-1)  # returns first occurrence
  # `eos_idx` is 0 if EOS is not present, so we use full length in that case.
  return jnp.where(
      eos_row[jnp.arange(eos_row.shape[0]), eos_idx],
      eos_idx + 1,
      sequence_batch.shape[1]  # if there is no EOS, use full length
  )
