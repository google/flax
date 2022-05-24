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
from typing import Any, Dict, Generator, Optional, Tuple

import jax.numpy as jnp
import numpy as np

Array = Any    # pylint: disable=invalid-name


class CharacterTable:
  """Encodes/decodes between strings and integer representations."""

  def __init__(self, chars: str, max_len_query_digit: int = 3) -> None:
    self._chars = sorted(set(chars))
    self._char_indices = {
        ch: idx + 2 for idx, ch in enumerate(self._chars)}
    self._indices_char = {
        idx + 2: ch for idx, ch in enumerate(self._chars)}
    self._indices_char[self.pad_id] = '_'
    # Maximum length of a single input digit.
    self._max_len_query_digit = max_len_query_digit

  @property
  def pad_id(self) -> int:
    return 0

  @property
  def eos_id(self) -> int:
    return 1

  @property
  def vocab_size(self) -> int:
    # All characters + pad token and eos token.
    return len(self._chars) + 2

  @property
  def max_input_len(self) -> int:
    """Returns the max length of an input sequence."""
    # The input has the form "digit1+digit2<eos>", so the max input length is
    # the length of two digits plus two tokens for "+" and the EOS token.
    return self._max_len_query_digit * 2 + 2

  @property
  def max_output_len(self) -> int:
    """Returns the max length of an output sequence."""
    # The output has the form "=digit<eos>". If `digit` is the result of adding
    # two digits of max length x, then max length of `digit` is x+1.
    # Additionally, we require two more tokens for "=" and "<eos".
    return self._max_len_query_digit + 3

  @property
  def encoder_input_shape(self) -> Tuple[int, int, int]:
    return (1, self.max_input_len, self.vocab_size)

  @property
  def decoder_input_shape(self) -> Tuple[int, int, int]:
    return (1, self.max_output_len, self.vocab_size)

  def encode(self, inputs: str) -> np.ndarray:
    """Encodes from string to list of integers."""
    return np.array(
        [self._char_indices[char] for char in inputs] + [self.eos_id])

  def decode(self, inputs: Array) -> str:
    """Decodes from list of integers to string."""
    chars = []
    for elem in inputs.tolist():
      if elem == self.eos_id:
        break
      chars.append(self._indices_char[elem])
    return ''.join(chars)

  def one_hot(self, tokens: np.ndarray) -> np.ndarray:
    vecs = np.zeros((tokens.size, self.vocab_size), dtype=np.float32)
    vecs[np.arange(tokens.size), tokens] = 1
    return vecs

  def encode_onehot(
      self, batch_inputs: Array, max_len: Optional[int] = None) -> np.ndarray:
    """One-hot encodes a string input."""

    if max_len is None:
      max_len = self.max_input_len

    def encode_str(s):
      tokens = self.encode(s)
      unpadded_len = len(tokens)
      if unpadded_len > max_len:
        raise ValueError(
            f'Sequence too long ({len(tokens)}>{max_len}): \'{s}\'')
      tokens = np.pad(tokens, [(0, max_len - len(tokens))], mode='constant')
      return self.one_hot(tokens)

    return np.array([encode_str(inp) for inp in batch_inputs])

  def decode_onehot(self, batch_inputs: Array) -> np.ndarray:
    """Decodes a batch of one-hot encoding to strings."""
    decode_inputs = lambda inputs: self.decode(inputs.argmax(axis=-1))
    return np.array(list(map(decode_inputs, batch_inputs)))

  def generate_examples(
      self, num_examples: int) -> Generator[Tuple[str, str], None, None]:
    """Yields `num_examples` examples."""
    for _ in range(num_examples):
      max_digit = pow(10, self._max_len_query_digit) - 1
      # TODO(marcvanzee): Use jax.random here.
      key = tuple(sorted((random.randint(0, 99), random.randint(0, max_digit))))
      inputs = f'{key[0]}+{key[1]}'
      # Preprend output by the decoder's start token.
      outputs = '=' + str(key[0] + key[1])
      yield (inputs, outputs)

  def get_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
    """Returns a batch of example of size @batch_size."""
    inputs, outputs = zip(*self.generate_examples(batch_size))
    return {
        'query': self.encode_onehot(inputs),
        'answer': self.encode_onehot(outputs),
    }


def mask_sequences(sequence_batch: Array, lengths: Array) -> Array:
  """Sets positions beyond the length of each sequence to 0."""
  return sequence_batch * (
      lengths[:, np.newaxis] > np.arange(sequence_batch.shape[1])[np.newaxis])


def get_sequence_lengths(sequence_batch: Array, eos_id: int) -> Array:
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
