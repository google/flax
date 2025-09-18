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

"""Input pipeline for a LM1B dataset."""

import dataclasses
import typing
import os

import grain
import jax
import numpy as np
import tensorflow_datasets as tfds
import tokenizer
from grain.python import MapTransform, MultiprocessingOptions

if typing.TYPE_CHECKING:
  from train import TrainConfig



Features = dict[str, np.ndarray]


class NormalizeFeatureNamesOp:
  """Normalizes feature names to 'inputs' and 'targets'."""

  def __call__(self, features: Features) -> Features:
    features['inputs'] = features.pop('text')
    features['targets'] = features['inputs']
    return features


def get_raw_dataset(dataset_name: str, split: str) -> grain.MapDataset:
  """Loads a raw text dataset and normalizes feature keys.

  Args:
    dataset_name: TFDS dataset name.
    split: Split to use. This must be the full split. We shard the split across
      multiple hosts and currently don't support sharding subsplits.

  Returns:
    Dataset with source and target language features mapped to 'inputs' and
    'targets'.
  """
  per_host_split = tfds.split_for_jax_process(split, drop_remainder=False)
  tfds_data_source = tfds.data_source(dataset_name, split=per_host_split)
  dataset = grain.MapDataset.source(tfds_data_source)
  dataset = dataset.map(NormalizeFeatureNamesOp())
  return dataset


# Code taken from MaxText
# https://github.com/AI-Hypercomputer/maxtext/blob/3a83b61afd894fbeb03ff48ec4993f39839afa9f/MaxText/input_pipeline/_input_pipeline_utils.py#L334-L335
@dataclasses.dataclass
class Rekey(MapTransform):
  """Rename keys according to a mapping dict"""

  def __init__(self, mapping_dict, keep_old_keys=False):
    self.mapping_dict = mapping_dict
    self.keep_old_keys = keep_old_keys

  def map(self, element):
    old_keys = set()
    for new_key, old_key in self.mapping_dict.items():
      element[new_key] = element[old_key]
      old_keys.add(old_key)
    if not self.keep_old_keys:
      for key in old_keys:
        del element[key]
    return element


def padded_batch(
    dataset: grain.IterDataset,
    batch_size: int,
    padded_size: int,
    padding_value: int,
    drop_remainder: bool,
):
  def padded_batching(values: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    def map_fn(*arrays):
      padded_arrays = [
        np.pad(
          array,
          (0, padded_size - array.shape[0]),
          'constant',
          constant_values=padding_value,
        ) for array in arrays
      ]
      return np.stack(padded_arrays)
    return jax.tree.map(map_fn, *values)
  return dataset.batch(batch_size, batch_fn=padded_batching, drop_remainder=drop_remainder)


def shift_target_left(x: dict[str, np.ndarray], pad_id: int) -> dict[str, np.ndarray]:
  # Shift to the left and pad the target
  targets = x["targets"][..., 1:]
  targets = np.pad(targets, [(0, 0), (0, 1)], "constant", constant_values=pad_id)
  x["targets"] = targets

  if "targets_segmentation" in x:
    # Set segmentation value associated to the added pad value to zero
    x["targets_segmentation"][..., -1] = 0
  return x


def move_to_devices(
  x: dict[str, np.ndarray], data_sharding: jax.sharding.Sharding
) -> dict[str, jax.Array]:
  return jax.tree.map(
    lambda value: jax.make_array_from_process_local_data(data_sharding, value), x
  )

# -----------------------------------------------------------------------------
# Main dataset prep routines.
# -----------------------------------------------------------------------------
def preprocess_data(
  dataset: grain.MapDataset,
  shuffle: bool,
  num_epochs: int | None = 1,
  pack_examples: bool = True,
  max_length: int = 512,
  batch_size: int = 256,
  drop_remainder: bool = True,
  shift: bool = True,
  seed: int = 41,
  prefetch_num_workers: int | None = None,
  pad_id: int = 0,
  data_sharding: jax.sharding.Sharding | None = None,
) -> grain.IterDataset:
  """Shuffle and batch/pack the given dataset."""

  def length_filter(max_len):
    def filter_fn(x):
      source, target = x['inputs'], x['targets']
      l = max(len(source), len(target))
      return l < max_len + 1
    return filter_fn

  if max_length > 0:
    dataset = dataset.filter(length_filter(max_length))

  if shuffle:
    dataset = dataset.shuffle(seed=seed)

  dataset = dataset.repeat(num_epochs)
  dataset = dataset.to_iter_dataset()

  if pack_examples:
    length_struct = {k: max_length for k in ["inputs", "targets"]}
    # grain.experimental.FirstFitPackIterDataset implicitly assumes pad_id=0
    # as it inserts token sequences to a zero array
    dataset = grain.experimental.FirstFitPackIterDataset(
        dataset, length_struct=length_struct, num_packing_bins=30
    )
    rekey_dict = {
        "targets_segmentation": "targets_segment_ids",
        "inputs_segmentation": "inputs_segment_ids",
        "targets_position": "targets_positions",
        "inputs_position": "inputs_positions",
    }
    dataset = dataset.map(Rekey(rekey_dict))
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  else:  # simple (static-shape) padded batching
    dataset = padded_batch(
      dataset,
      batch_size,
      padded_size=max_length,
      padding_value=pad_id,
      drop_remainder=drop_remainder,
    )

  # Shift inputs for teacher-forced training
  if shift:
    # We shift after packing:
    # pad_id, bos_id, eos_id, unk_id = -1, 1, 2, 0
    #  inputs:  1  t1 t2 t3 2 | 1  T1 T2 2 | 0  0
    # targets:  t1 t2 t3 2  1 | T1 T2 2  0 | 0 -1
    dataset = dataset.map(lambda x: shift_target_left(x, pad_id=pad_id))

  if prefetch_num_workers is None:
    prefetch_num_workers = min(os.cpu_count() // 2, 32)

  dataset = dataset.mp_prefetch(MultiprocessingOptions(num_workers=prefetch_num_workers))

  # Move data to jax array
  if data_sharding is not None:
    dataset = dataset.map(lambda x: move_to_devices(x, data_sharding=data_sharding))
  else:
    dataset = dataset.map(lambda x: jax.tree.map(jax.numpy.asarray, x))

  return dataset


def get_datasets(
  config: "TrainConfig",
  *,
  vocab_path: str | None = None,
  data_sharding: jax.sharding.Sharding | None = None,
):
  """Load and return dataset of batched examples for use during training."""
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/lm1b_sentencepiece_model')

  train_data = get_raw_dataset(config.dataset_name, split="train")
  eval_data = get_raw_dataset(config.eval_dataset_name, split=config.eval_split)

  # Tokenize data.
  sp_processor = tokenizer.load_or_train_tokenizer(
    train_data,
    vocab_path=vocab_path,
    vocab_size=config.vocab_size,
    max_corpus_chars=config.max_corpus_chars,
  )
  train_data = train_data.map(tokenizer.TokenizeOp(sp_processor))
  eval_data = eval_data.map(tokenizer.TokenizeOp(sp_processor))

  if data_sharding is not None:
    n_devices = len(data_sharding.device_set)
  else:
    n_devices = 1

  batch_size = config.per_device_batch_size * n_devices
  if config.eval_per_device_batch_size > 0:
    eval_batch_size = config.eval_per_device_batch_size * n_devices
  else:
    eval_batch_size = batch_size

  train_ds = preprocess_data(
    train_data,
    shuffle=True,
    num_epochs=None,
    pack_examples=True,
    batch_size=batch_size,
    max_length=config.max_target_length,
    seed=config.seed,
    prefetch_num_workers=config.prefetch_num_workers,
    pad_id=sp_processor.pad_id(),
    data_sharding=data_sharding,
  )

  eval_ds = preprocess_data(
    eval_data,
    shuffle=False,
    pack_examples=False,
    batch_size=eval_batch_size,
    max_length=config.max_eval_target_length,
    prefetch_num_workers=config.prefetch_num_workers,
    pad_id=sp_processor.pad_id(),
    data_sharding=data_sharding,
  )

  return train_ds, eval_ds, sp_processor
