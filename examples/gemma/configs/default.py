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

"""Default Hyperparameter configuration."""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(unsafe_hash=True)
class MeshRules:
  embed: str | None = None
  mlp: str | None = None
  kv: str | None = None
  vocab: str | None = None

  def __call__(self, *keys: str) -> tuple[str, ...]:
    return tuple(getattr(self, key) for key in keys)


@dataclasses.dataclass(unsafe_hash=True)
class Config:
  # Path to load or store sentencepiece vocab file.
  vocab_path: str | None = None
  # Vocabulary size if `vocab_path` is not given.
  vocab_size: int = 30_000  # lm1b dataset vocab size: 35913  (Gemma expected vocab size: 262_144)
  # Maximum number of characters to use for training.
  max_corpus_chars: int = 10**7
  # Name of TFDS translation dataset to use.
  dataset_name: str = 'lm1b'
  # Optional name of TFDS translation dataset to use for evaluation.
  eval_dataset_name: str = 'lm1b'
  # Optional name of TFDS split to use for evaluation.
  eval_split: str = 'test'
  # Per device batch size for training.
  per_device_batch_size: int = 32
  # Per device batch size for training.
  eval_per_device_batch_size: int = 32

  # Prompt for language model sampling
  prompts: tuple[str, ...] = ('Paris is a the capital', 'Flax is a')
  # Temperature for top_p sampling.
  sampling_temperature: float = 5.0
  # Top-p sampling threshold.
  sampling_top_p: float = 0.95

  # Number of steps to take during training.
  num_train_steps: int = 500_000
  # Number of steps to take during evaluation.
  # Large enough to evaluate all samples: 306_688 / (32 * 8) = 1198
  num_eval_steps: int = 2_000
  # Number of steps to generate predictions.
  # -1 will use the whole eval dataset.
  num_predict_steps: int = -1
  # Base learning rate.
  learning_rate: float = 0.0016
  # Linear learning rate warmup.
  warmup_steps: int = 1000
  # Cross entropy loss label smoothing.
  label_smoothing: float = 0.0
  # Decay factor for AdamW style weight decay.
  weight_decay: float = 0.1
  # Maximum length cutoff for training examples.
  max_target_length: int = 128
  # Maximum length cutoff for eval examples.
  max_eval_target_length: int = 512

  # Gemma transformer name.
  # Possible values defined in transformer.TransformerConfig:
  # (gemma_2b, gemma_7b, gemma2_2b, gemma2_9b, gemma2_27b, gemma3_1b, gemma3_4b, ...)
  transformer_name: str | None = "gemma3_1b"
  # or alternatively define the model using the dict of parameters
  transformer_params: dict | None = None

  # Whether to save model checkpoints.
  save_checkpoints: bool = True
  # Whether to restore from existing model checkpoints.
  restore_checkpoints: bool = True
  # Save a checkpoint every these number of steps.
  checkpoint_every_steps: int = 10_000
  # Frequency of eval during training, e.g. every 1_000 steps.
  eval_every_steps: int = 1_000
  # Use bfloat16 mixed precision training instead of float32.
  use_bfloat16: bool = True
  # Integer for PRNG random seed.
  seed: int = 0

  # Parallelism
  mesh_axes: tuple[str, ...] = ('data', 'fsdp', 'tensor')
  axis_rules: MeshRules = MeshRules(
    embed='fsdp',
    mlp='tensor',
    kv='tensor',
    vocab='tensor',
  )
  data_sharding: tuple[str, ...] = ('data',)

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
  dcn_data_parallelism: int = -1
  dcn_fsdp_parallelism: int = 1
  dcn_tensor_parallelism: int = 1
  ici_data_parallelism: int = 1
  ici_fsdp_parallelism: int = -1
  ici_tensor_parallelism: int = 1

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)


def get_config():
  """Get the default hyperparameter configuration."""
  config = Config()
  return config
