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

"""Provides op for tokenizing a dataset."""

import dataclasses
import os
import tempfile
import time
from typing import Any
from collections.abc import Iterable

import jax
import tensorflow as tf
import tensorflow_text as tftxt
from absl import logging
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

Features = dict[str, tf.Tensor]


def _dump_chars_to_textfile(
  dataset: tf.data.Dataset,
  maxchars: int = int(1e7),
  data_keys=('inputs', 'targets'),
) -> tuple[str, int]:
  """Write part of a TFDS sentence dataset to lines in a text file.

  Args:
    dataset: tf.dataset containing string-data.
    maxchars: int: approximate number of characters to save from dataset.
    data_keys: Tuple[str]: what keys in dataset to dump from.

  Returns:
    name of temp file with dataset bytes, exact number of characters dumped.
  """
  char_count = 0
  ds_iter = dataset.as_numpy_iterator()
  with tempfile.NamedTemporaryFile(
    delete=False, prefix='/tmp/ds_chars'
  ) as outfp:
    while char_count < maxchars:
      example = next(ds_iter)
      for k in data_keys:
        line = example[k] + b'\n'
        char_count += len(line)
        outfp.write(line)
  return outfp.name, char_count


def _train_sentencepiece(
  dataset: tf.data.Dataset,
  *,
  vocab_size: int,
  maxchars: int = int(1e7),
  model_path: str,
  model_type: str = 'unigram',
  character_coverage: float = 1.0,
  data_keys=('inputs', 'targets'),
  pad_id: int = 0,
  eos_id: int = 1,
  bos_id: int = 2,
  unk_id: int = 3,
):
  """Train SentencePiece tokenizer from subset of tf dataset.

  Args:
    dataset: tf.dataset
    vocab_size: int: size of vocab tokens to train.
    maxchars: int: number of characters to use for sentencepiece training.
    model_path: str: path of model file to save vocab model to.
    model_type: str: type of sentencepiece vocab to train.
    character_coverage: amount of characters covered by the model, good defaults
      are 0.9995 for languages with rich character set like Japanese or Chinese
      and 1.0 for other languages with small character set.
    data_keys: Tuple[str]: keys of dataset to use for training.
    pad_id: int: pad piece id
    eos_id: int: end of sentence piece id
    bos_id: int: begin of sentence piece id
    unk_id: int: unknown piece id

  Returns:
    path to the trained sentencepiece vocabulary model.
  """
  if model_path.startswith('gs://'):
    abs_model_path = model_path
  else:
    abs_model_path = os.path.abspath(os.path.expanduser(model_path))
  fname, _ = _dump_chars_to_textfile(
    dataset, maxchars=maxchars, data_keys=data_keys
  )
  with tempfile.NamedTemporaryFile(
    delete=False, prefix='/tmp/sp_tmp'
  ) as model_fp:
    pass  # we just want a prefix'd tmp-filename
  argstr = ' '.join(
    [
      f'--input={fname}',
      f'--vocab_size={vocab_size}',
      f'--character_coverage={character_coverage}',
      f'--model_prefix={model_fp.name}',
      f'--model_type={model_type}',
      # Setup ids for PAD, EOS, BOS, UNK as 0, 1, 2, 3
      # Default values:
      # --unk_id (Override UNK (<unk>) id.)  type: int32 default: 0
      # --bos_id (Override BOS (<s>) id. Set -1 to disable BOS.)  type: int32 default: 1
      # --eos_id (Override EOS (</s>) id. Set -1 to disable EOS.)  type: int32 default: 2
      # --pad_id (Override PAD (<pad>) id. Set -1 to disable PAD.)  type: int32 default: -1
      # https://github.com/google/sentencepiece/blob/master/doc/options.md
      f'--pad_id={pad_id}',
      f'--bos_id={bos_id}',
      f'--eos_id={eos_id}',
      f'--unk_id={unk_id}',
    ]
  )
  SentencePieceTrainer.Train(argstr)
  if jax.process_index() == 0:
    # Use an intermediate filename that is renamed to the target name to address
    # create and fill delays.
    copy_rename_path = abs_model_path + '.rntmp'
    tf.io.gfile.copy(model_fp.name + '.model', copy_rename_path, overwrite=True)
    tf.io.gfile.rename(copy_rename_path, abs_model_path, overwrite=True)
    logging.info('copied %s to %s', model_fp.name + '.model', abs_model_path)
  else:
    while not tf.io.gfile.exists(abs_model_path):
      time.sleep(1)
    time.sleep(1)
  return abs_model_path


def _load_sentencepiece_tokenizer(
  model_path: str,
  add_bos: bool = False,
  add_eos: bool = True,
  reverse: bool = False,
):
  """Load a tf-text SentencePiece tokenizer from given model filepath."""
  with tf.io.gfile.GFile(model_path, 'rb') as model_fp:
    sp_model = model_fp.read()
  sp_tokenizer = tftxt.SentencepieceTokenizer(
    model=sp_model, add_bos=add_bos, add_eos=add_eos, reverse=reverse
  )
  return sp_tokenizer


def load_or_train_tokenizer(
  dataset: tf.data.Dataset,
  *,
  vocab_path: str,
  vocab_size: int,
  max_corpus_chars: int,
  data_keys: tuple[str, str] = ('inputs', 'targets'),
):
  """Loads the tokenizer at `vocab_path` or trains a one from `dataset`."""
  try:
    return _load_sentencepiece_tokenizer(vocab_path)
  except tf.errors.NotFoundError:
    logging.info('SentencePiece vocab not found, building one from data.')
    vocab_path = _train_sentencepiece(
      dataset,
      vocab_size=vocab_size,
      maxchars=max_corpus_chars,
      model_path=vocab_path,
      data_keys=data_keys,
    )
    return _load_sentencepiece_tokenizer(vocab_path)


@dataclasses.dataclass
class TokenizeOp:
  sp_tokenizer: Any
  data_keys: Iterable[str] = ('inputs', 'targets')

  def __call__(self, features: Features) -> Features:
    for k in self.data_keys:
      features[k] = self.sp_tokenizer.tokenize(features[k])
    return features


def load_sentencepiece_processor(vocab_path: str):
  spp = SentencePieceProcessor()
  spp.load(vocab_path)
  return spp
