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

"""A vocabulary that represents the tokens in a dataset and maps them to indices."""

import collections
from typing import Iterable, Optional, Sequence

from absl import logging


class Vocabulary:
  """Represents a vocabulary that can be built from a dataset."""

  def __init__(self,
               vocab_path: Optional[str] = None,
               tokenized_sequences: Optional[Iterable[Sequence[bytes]]] = None,
               min_freq: int = 1,
               pad_token: bytes = b'<pad>',
               unk_token: bytes = b'<unk>',
               bos_token: bytes = b'<s>',
               eos_token: bytes = b'</s>'):
    """Loads the vocab from disk (if `vocab_path` is given) or builds it from `tokenized_sequences`."""
    self.pad_token = pad_token
    self.unk_token = unk_token
    self.bos_token = bos_token
    self.eos_token = eos_token
    self.special_tokens = (pad_token, unk_token, bos_token, eos_token)

    if vocab_path:
      self.load(vocab_path)
    elif tokenized_sequences is not None:
      self.build(tokenized_sequences, min_freq=min_freq)
    else:
      raise ValueError(
          ('Vocabulary needs either `vocab_path` or `tokenized_sequences` to '
           'be provided, got %r and %r.') % (vocab_path, tokenized_sequences))

  def build(self,
            tokenized_sequences: Iterable[Sequence[bytes]],
            min_freq: int = 1):
    """Builds a vocabulary over tokens with optional minimum frequency.

    Args:
      tokenized_sequences: Iterable of token sequences to build the vocabulary.
      min_freq: The minimum frequency of each token to be included. Default: 1.
    """
    # Count all the tokens.
    counter = collections.Counter()
    for token_sequence in tokenized_sequences:
      counter.update(token_sequence)

    # Add special tokens to the start of vocab.
    vocab = collections.OrderedDict()
    for token in self.special_tokens:
      vocab[token] = len(vocab)

    # Add all other tokens to the vocab if their frequency is >= min_freq.
    for token, freq in sorted(
        # Sort by frequency (from high to low), and then by token string.
        # This makes sure high frequency tokens get a low token ID.
        counter.items(),
        key=lambda token_freq: (-token_freq[1], token_freq[0])):
      if freq >= min_freq:
        vocab[token] = len(vocab)

    logging.info('Number of unfiltered tokens: %d', len(counter))
    logging.info('Vocabulary size: %d', len(vocab))
    self.vocab = vocab

  def _getitem__(self, key: str):
    return self.vocab[key]

  def keys(self):
    return self.vocab.keys()

  def values(self):
    return self.vocab.values()

  def __len__(self):
    return len(self.vocab)

  @property
  def pad_idx(self):
    """The index of the padding token."""
    return self.vocab[self.pad_token]

  @property
  def unk_idx(self):
    """The index of the unknown word token."""
    return self.vocab[self.unk_token]

  @property
  def bos_idx(self):
    """The index of the beginning-of-sequence token."""
    return self.vocab[self.bos_token]

  @property
  def eos_idx(self):
    """The index of the end-of-sequence token."""
    return self.vocab[self.eos_token]

  def load(self, path: str) -> None:
    """Loads a vocabulary (one token per line) from disk."""
    vocab = collections.OrderedDict()
    with open(path, 'rb') as f:
      for i, token in enumerate(f):
        assert isinstance(token, bytes), 'Expected byte tokens.'
        vocab[token.rstrip()] = i
    logging.info('Loaded vocabulary, size: %d', len(vocab))
    self.vocab = vocab

  def save(self, path: str) -> None:
    """Saves the vocabulary to disk."""
    with open(path, 'wb') as f:
      for token in self.vocab:
        assert isinstance(token, bytes), 'Expected byte tokens.'
        f.write(token + b'\n')
    logging.info('Saved vocabulary to %s.', path)
