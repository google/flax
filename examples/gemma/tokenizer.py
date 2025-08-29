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
import tempfile
import time
import shutil
from pathlib import Path
from typing import Any, Iterable
from collections.abc import Iterable

import jax
import numpy as np
from absl import logging
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor


def _dump_chars_to_textfile(
  dataset: Iterable,
  maxchars: int = int(1e7),
  data_keys=('inputs', 'targets'),
) -> tuple[str, int]:
  """Write part of a TFDS sentence dataset to lines in a text file.

  Args:
    dataset: Grain dataset containing string-data.
    maxchars: int: approximate number of characters to save from dataset.
    data_keys: Tuple[str]: what keys in dataset to dump from.

  Returns:
    name of temp file with dataset bytes, exact number of characters dumped.
  """
  char_count = 0
  ds_iter = iter(dataset)
  with tempfile.NamedTemporaryFile(
    mode="w+b", delete=False, prefix='/tmp/ds_chars'
  ) as outfp:
    while char_count < maxchars:
      example = next(ds_iter)
      for k in data_keys:
        data = example[k]
        line = (data.encode() if isinstance(data, str) else data) + b'\n'
        char_count += len(line)
        outfp.write(line)
  return outfp.name, char_count


def _train_sentencepiece(
  dataset: Any,
  *,
  vocab_size: int,
  maxchars: int = int(1e7),
  model_path: str,
  model_type: str = 'unigram',
  character_coverage: float = 1.0,
  data_keys=('inputs', 'targets'),
  pad_id: int = 0,
  unk_id: int = 3,
  bos_id: int = 2,
  eos_id: int = 1,
):
  """Train SentencePiece tokenizer from subset of tf dataset.

  Args:
    dataset: Grain dataset
    vocab_size: int: size of vocab tokens to train.
    maxchars: int: number of characters to use for sentencepiece training.
    model_path: str: path of model file to save vocab model to.
    model_type: str: type of sentencepiece vocab to train.
    character_coverage: amount of characters covered by the model, good defaults
      are 0.9995 for languages with rich character set like Japanese or Chinese
      and 1.0 for other languages with small character set.
    data_keys: Tuple[str]: keys of dataset to use for training.
    pad_id: int: pad piece id
    unk_id: int: unknown piece id
    bos_id: int: begin of sentence piece id
    eos_id: int: end of sentence piece id
  Returns:
    path to the trained sentencepiece vocabulary model.
  """
  model_path = Path(model_path)
  abs_model_path = model_path.expanduser().absolute().resolve()
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
    copy_rename_path = abs_model_path.with_suffix('.rntmp')
    shutil.copyfile(model_fp.name + '.model', copy_rename_path)
    shutil.move(copy_rename_path, abs_model_path)
    logging.info('copied %s to %s', model_fp.name + '.model', abs_model_path)
  else:
    while not abs_model_path.exists():
      time.sleep(1)
    time.sleep(1)
  return str(abs_model_path)


def load_or_train_tokenizer(
  dataset: Any,
  *,
  vocab_path: str,
  vocab_size: int,
  max_corpus_chars: int,
  data_keys: tuple[str, str] = ('inputs', 'targets'),
):
  """Loads the tokenizer at `vocab_path` or trains a one from `dataset`."""
  try:
    return load_sentencepiece_processor(vocab_path)
  except (OSError, TypeError):
    logging.info('SentencePiece vocab not found, building one from data.')
    vocab_path = _train_sentencepiece(
      dataset,
      vocab_size=vocab_size,
      maxchars=max_corpus_chars,
      model_path=vocab_path,
      data_keys=data_keys,
    )
    return load_sentencepiece_processor(vocab_path)


@dataclasses.dataclass
class TokenizeOp:
  sp_processor: SentencePieceProcessor
  data_keys: Iterable[str] = ('inputs', 'targets')

  def __call__(self, features: dict[str, str]) -> dict[str, np.ndarray]:
    for k in self.data_keys:
      features[k] = np.array(
        self.sp_processor.EncodeAsIds(features[k], add_eos=True, add_bos=True), dtype=np.int32
      )
    return features


def load_sentencepiece_processor(vocab_path: str):
  spp = SentencePieceProcessor()
  spp.load(vocab_path)
  return spp
