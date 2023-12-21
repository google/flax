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

"""Default Hyperparameter configuration."""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Config:
  vocab_path: str | None
  vocab_size: int
  max_corpus_chars: int
  dataset_name: str
  eval_dataset_name: str
  eval_split: str
  per_device_batch_size: int
  eval_per_device_batch_size: int
  sampling_temperature: float
  sampling_top_k: int
  num_train_steps: int
  num_eval_steps: int
  num_predict_steps: int
  learning_rate: float
  warmup_steps: int
  label_smoothing: float
  weight_decay: float
  max_target_length: int
  max_eval_target_length: int
  max_predict_length: int
  logits_via_embedding: bool
  num_layers: int
  qkv_dim: int
  emb_dim: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float
  attention_dropout_rate: float
  save_checkpoints: bool
  restore_checkpoints: bool
  checkpoint_every_steps: int
  eval_every_steps: int
  use_bfloat16: bool
  seed: int
  prompts: str
  mesh_axes: list[str]
  logical_axis_rules: list[list[str | list[str]] | list[str]]
  full_sharding: list[str]
  data_sharding: list[str]
  dcn_data_parallelism: int
  dcn_fsdp_parallelism: int
  dcn_tensor_parallelism: int
  ici_data_parallelism: int
  ici_fsdp_parallelism: int
  ici_tensor_parallelism: int

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)


def get_config():
  """Get the default hyperparameter configuration."""
  config = Config(
    # Path to load or store sentencepiece vocab file.
    vocab_path=None,
    # Vocabulary size if `vocab_path` is not given.
    vocab_size=30_000,
    max_corpus_chars=10**7,
    # Name of TFDS translation dataset to use.
    dataset_name='lm1b',
    # Optional name of TFDS translation dataset to use for evaluation.
    eval_dataset_name='lm1b',
    eval_split='test',
    # Per device batch size for training.
    per_device_batch_size=32,
    # Per device batch size for training.
    eval_per_device_batch_size=32,
    # Sampling temperature for language model inference.
    sampling_temperature=0.6,
    # Top k cutoff for logit sampling. If 0 then no top-k cutoff is used.
    sampling_top_k=20,
    num_train_steps=500_000,
    # Number of steps to take during evaluation. Large enough to evaluate all.
    # Large enough to evaluate all samples: 306_688 / (32 * 8) = 1198
    num_eval_steps=2_000,
    # Number of steps to generate predictions.
    # -1 will use the whole eval dataset.
    num_predict_steps=-1,
    # Base learning rate.
    learning_rate=0.0016,
    # Linear learning rate warmup.
    warmup_steps=1000,
    # Cross entropy loss label smoothing.
    label_smoothing=0.0,
    # Decay factor for AdamW style weight decay.
    weight_decay=0.1,
    # Maximum length cutoff for training examples.
    max_target_length=128,
    # Maximum length cutoff for eval examples.
    max_eval_target_length=512,
    # Maximum length cutoff for predicted tokens.
    max_predict_length=50,
    # Final logit transform uses embedding matrix transpose.
    logits_via_embedding=False,
    # Number of transformer layers.
    num_layers=6,
    # Size of query/key/value for attention.
    qkv_dim=512,
    # Size of embeddings.
    emb_dim=512,
    # Size of the MLP.
    mlp_dim=2048,
    # Number of attention heads.
    num_heads=8,
    # Dropout rate.
    dropout_rate=0.1,
    # Attention dropout rate.
    attention_dropout_rate=0.1,
    # Whether to save model checkpoints.
    save_checkpoints=True,
    # Whether to restore from existing model checkpoints.
    restore_checkpoints=True,
    # Save a checkpoint every these number of steps.
    checkpoint_every_steps=10_000,
    # Frequency of eval during training, e.g. every 1_000 steps.
    eval_every_steps=1_000,
    # Use bfloat16 mixed precision training instead of float32.
    use_bfloat16=True,
    # Integer for PRNG random seed.
    seed=0,
    # Prompt for language model sampling,
    # taken from MaxText (https://github.com/google/maxtext/blob/main/MaxText/configs/base.yml).
    prompts='I love to ',
    # Parallelism
    mesh_axes=['data', 'fsdp', 'tensor'],
    logical_axis_rules=[
      ['activation_batch', ['data', 'fsdp']],
      ['activation_length', ['data', 'fsdp']],
      ['activation_embed', 'tensor'],
      ['activation_mlp', 'tensor'],
      ['activation_heads', 'tensor'],
      ['activation_kv', 'tensor'],
      ['activation_vocab', 'tensor'],
      ['mlp', 'tensor'],
      ['vocab', 'tensor'],
      ['embed', 'fsdp'],
      ['heads', 'tensor'],
    ],
    full_sharding=['data', 'fsdp', 'tensor'],
    data_sharding=['data'],
    # One axis for each parallelism type may hold a placeholder (-1)
    # value to auto-shard based on available slices and devices.
    # By default, product of the DCN axes should equal number of slices
    # and product of the ICI axes should equal number of devices per slice.
    # ICI (Inter-Chip Interconnection): A high-speed connection between
    # sets of TPU chips, which form the TPU network.
    # DCN (Data Center Network): A connection between the TPU networks;
    # not as fast as ICI.
    # ICI has around 100x the bandwidth of DCN, but it is not a general
    # purpose connection, which is why DCN is necessary for scaling to
    # extremely large ML models.
    dcn_data_parallelism=-1,  # recommended DCN axis to be auto-sharded
    dcn_fsdp_parallelism=1,
    dcn_tensor_parallelism=1,
    ici_data_parallelism=1,
    ici_fsdp_parallelism=-1,  # recommended ICI axis to be auto-sharded
    ici_tensor_parallelism=1,
  )

  return config
